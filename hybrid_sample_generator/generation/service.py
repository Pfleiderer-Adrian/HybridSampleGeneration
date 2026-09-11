"""Application service for training, loading and running generator models."""

from __future__ import annotations

import optuna
import numpy as np
import torch
from tqdm import tqdm

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from hybrid_sample_generator.domain.records import SyntheticAnomaly
from hybrid_sample_generator.generation.interfaces import GenerativeBackend
from hybrid_sample_generator.generation.registry import get_model_spec
from hybrid_sample_generator.generation.training.optuna import optimize
from hybrid_sample_generator.imaging.masks.transform_generator import (
    TransformGenerator,
)
from hybrid_sample_generator.imaging.similarity import ssim_01
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.identifiers import stable_id, stable_seed
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.randomness import seeded_random


class GenerationService:
    """Own the active generator model and its record-based workflows."""

    def __init__(
        self,
        config: Configuration,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        datasets: StudyDatasets,
        *,
        model: GenerativeBackend | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.artifact_store = artifact_store
        self.datasets = datasets
        self._model = model

    @property
    def model(self) -> GenerativeBackend | None:
        return self._model

    def train(self, no_of_trials: int) -> None:
        """Optimize generator hyperparameters using persisted real anomalies."""
        optimize(no_of_trials, self.config, self._training_dataset())

    def load(self, path_to_db_file=None, trial_id: int = -1) -> GenerativeBackend:
        """Build and load the generator referenced by an Optuna trial."""
        storage = (
            self.config.study.paths.optuna_storage_url
            if path_to_db_file is None
            else "sqlite:///" + str(path_to_db_file)
        )
        study = optuna.load_study(
            study_name=self.config.study.name,
            storage=storage,
        )
        trial = _select_trial(study, trial_id)

        model = get_model_spec(trial.user_attrs["model_name"]).build(
            trial.user_attrs["params"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.warmup(
            self.config.extraction.anomaly_size,
            device=device,
            dtype=self.config.training.dtype,
            config=self.config.training,
        )
        model.load_checkpoint(trial.user_attrs["model_path"])
        self._model = model
        return model

    def generate(self) -> list[SyntheticAnomaly]:
        """Generate and persist configured variants for every real anomaly."""
        if self._model is None:
            raise ValueError("No generator model loaded.")

        source_dataset = self.datasets.real_anomalies(
            return_artifacts=("img", "fname", "ori_mask", "real_anomaly_id"),
            load_to_ram=False,
            numpy_mode=True,
        )
        if not len(source_dataset):
            raise ValueError("No real anomalies found. Run extract_anomalies first.")

        self.repository.clear_synthetic_and_downstream()
        generated: list[SyntheticAnomaly] = []
        generation = self.config.generation

        for sample in tqdm(source_dataset):
            real_id = sample["real_anomaly_id"]
            real_image = sample["img"]
            for variant_index in range(int(generation.variants_per_real_anomaly)):
                seed = stable_seed(self.config.study.seed, real_id, variant_index)
                target_mask_generator = TransformGenerator.from_config(
                    self.config.augmentation.mask_transforms,
                    anomaly_size=self.config.extraction.anomaly_size,
                    background_threshold=generation.background_threshold,
                    seed=seed,
                )
                with seeded_random(seed):
                    image, mask = self._generate_variant(
                        sample,
                        real_image,
                        target_mask_generator,
                    )

                image, mask = _validate_generated_variant(
                    image,
                    mask,
                    real_image,
                )
                synthetic_id = stable_id("synthetic", real_id, variant_index)
                image_path = self.artifact_store.save_entity_array(
                    "synthetic_anomalies",
                    synthetic_id,
                    "image",
                    image,
                )
                segmentation_path = self.artifact_store.save_entity_array(
                    "synthetic_anomalies",
                    synthetic_id,
                    "segmentation",
                    mask,
                )
                record = SyntheticAnomaly(
                    id=synthetic_id,
                    real_anomaly_id=real_id,
                    variant_index=variant_index,
                    image_path=image_path,
                    segmentation_path=segmentation_path,
                    seed=seed,
                )
                self.repository.upsert_synthetic_anomaly(record)
                generated.append(record)

        return generated

    def _training_dataset(self):
        records = self.repository.list_real_anomalies()
        if not records:
            raise ValueError("No real anomalies found. Run extract_anomalies first.")

        if get_model_spec(self.config.model.name).uses_masks:
            max_class = max(
                int(round(float(record.metadata.get("label", 0))))
                for record in records
            )
            self.config.model.parameters.set_model_param(
                "num_anomaly_classes",
                max_class,
            )

        return self.datasets.real_anomalies(
            return_artifacts=self.config.model.parameters.input_artefacts,
            load_to_ram=True,
            dtype=torch.float32,
        )

    def _generate_variant(
        self,
        sample,
        real_image,
        target_mask_generator: TransformGenerator,
    ):
        generation = self.config.generation
        kwargs = {
            "mode": generation.sampling_mode,
            "variation_strength": generation.variation_strength,
            "clamp_01": generation.clamp_output,
            "target_mask_generator": target_mask_generator,
        }

        if not generation.feedback.enabled:
            image, mask = self._model.generate(sample, **kwargs)
            return _as_numpy(image), _as_numpy(mask)

        best_score = -np.inf
        best = None
        threshold = float(generation.feedback.similarity_threshold)

        for attempt in range(int(generation.feedback.max_attempts)):
            image, mask = self._model.generate(sample, **kwargs)
            image, mask = _as_numpy(image), _as_numpy(mask)
            if image.shape != real_image.shape:
                raise ValueError(
                    f"Generated shape {image.shape} differs from {real_image.shape}."
                )

            score = ssim_01(real_image, image)
            if score > best_score:
                best_score = score
                best = image, mask
            if score >= threshold:
                break
            if (attempt + 1) % 100 == 0:
                threshold *= float(
                    generation.feedback.threshold_relaxation_factor
                )

        if best is None:
            raise RuntimeError("Generator produced no variant.")
        return best


def _select_trial(study, trial_id: int):
    if trial_id == -1:
        return study.best_trial

    trials = study.get_trials()
    if trial_id == -2:
        return max(trials, key=lambda value: value.number)

    trial = next(
        (value for value in trials if value.number == trial_id),
        None,
    )
    if trial is None:
        raise ValueError(f"Optuna trial {trial_id} does not exist.")
    return trial


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_generated_variant(image, mask, expected_image):
    image = _as_numpy(image)
    mask = _as_numpy(mask)
    expected_shape = np.asarray(expected_image).shape

    if image.shape != expected_shape:
        raise ValueError(
            f"Generated image shape {image.shape} differs from expected "
            f"{expected_shape}."
        )
    if (
        mask.ndim != image.ndim
        or mask.shape[1:] != expected_shape[1:]
        or mask.shape[0] not in (1, expected_shape[0])
    ):
        raise ValueError(
            "Generated mask must share the image's spatial shape and have one or "
            f"{expected_shape[0]} channels, got {mask.shape}."
        )
    return image, mask
