from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import torch
from tqdm import tqdm

from data_handler.StudyDatasets import StudyDatasets
from fusion_backend.interfaces import FusionBackend
from fusion_backend.fusion_registry import get_fusion_backend_spec
from generation_models.interfaces import GenerativeBackend
from generation_models.model_registry import get_model_spec
from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.Configuration import Configuration
from synthesizer.InputSample import iter_input_samples
from synthesizer.Matching import (
    combine_label_masks,
    persist_original_sample,
    plan_hybrid_samples as build_hybrid_plan,
    ssim_01,
)
from synthesizer.StudyRecords import HybridSample, RealAnomaly, SyntheticAnomaly
from synthesizer.StudyRepository import StudyRepository, stable_id, stable_seed
from synthesizer.Trainer import optimize
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_and_center_anomaly_2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_and_center_anomaly_3d
from synthesizer.mask_manipulation import TransformGenerator


class HybridDataGenerator:
    """Coordinates write operations for the record-based generation pipeline.

    Persisted state lives in StudyRepository. Only expensive runtime components
    (the generator model and fusion backend) are retained between method calls.
    """

    def __init__(
        self,
        config: Configuration,
        *,
        generator_model: GenerativeBackend | None = None,
        fusion_backend: FusionBackend | None = None,
    ) -> None:
        config.validate()
        self.config = config
        paths = config.study.paths
        self.repository = StudyRepository(paths.artifact_database)
        self.artifact_store = ArtifactStore(paths.study_folder)
        self.datasets = StudyDatasets(self.repository, self.artifact_store)
        self._generator_model = generator_model
        self._fusion_backend = fusion_backend

    def _log_step(self, message: str) -> None:
        print(f"[HybridDataGenerator] {message}")

    def extract_anomalies(self, sample_dataloader) -> list[RealAnomaly]:
        """Extract and register real anomalies and their owning original samples."""
        self._log_step("Extracting real anomalies into normalized study records.")
        self.repository.clear_all_records()
        extracted: list[RealAnomaly] = []

        for sample in iter_input_samples(sample_dataloader):
            if not np.any(sample.segmentation):
                continue
            original = persist_original_sample(sample, self.repository, self.artifact_store)
            if sample.image.ndim == 3:
                result = crop_and_center_anomaly_2d(
                    sample.image, sample.segmentation, self.config.extraction
                )
            elif sample.image.ndim == 4:
                result = crop_and_center_anomaly_3d(
                    sample.image, sample.segmentation, self.config.extraction
                )
            else:
                raise ValueError(
                    f"Unexpected shape {sample.image.shape}; expected (C,H,W) or (C,D,H,W)."
                )
            if not result or result[0] is None:
                continue
            anomalies, anomaly_rois, masks, roi_masks = result
            if not (len(anomalies) == len(anomaly_rois) == len(masks) == len(roi_masks)):
                raise RuntimeError("Extraction returned unaligned anomaly artifacts.")

            for component_index, ((image, metadata), roi, mask, roi_mask) in enumerate(
                zip(anomalies, anomaly_rois, masks, roi_masks)
            ):
                record_id = stable_id("real", original.id, component_index)
                image_path = self.artifact_store.save_entity_array(
                    "real_anomalies", record_id, "image", image
                )
                segmentation_path = self.artifact_store.save_entity_array(
                    "real_anomalies", record_id, "segmentation", mask
                )
                roi_image_path = self.artifact_store.save_entity_array(
                    "real_anomalies", record_id, "roi_image", roi
                )
                roi_segmentation_path = self.artifact_store.save_entity_array(
                    "real_anomalies", record_id, "roi_segmentation", roi_mask
                )
                position = tuple(float(value) for value in metadata["centroid_norm"])
                position_z, position_y, position_x = _position_columns(position)
                record = RealAnomaly(
                    id=record_id,
                    original_sample_id=original.id,
                    component_index=component_index,
                    image_path=image_path,
                    segmentation_path=segmentation_path,
                    roi_image_path=roi_image_path,
                    roi_segmentation_path=roi_segmentation_path,
                    spatial_dimensions=sample.image.ndim - 1,
                    position_z=position_z,
                    position_y=position_y,
                    position_x=position_x,
                    metadata=dict(metadata),
                )
                self.repository.upsert_real_anomaly(record)
                extracted.append(record)

        if not extracted:
            raise ValueError("No real anomalies were extracted from the supplied samples.")
        return extracted

    def _training_dataset(self):
        records = self.repository.list_real_anomalies()
        if not records:
            raise ValueError("No real anomalies found. Run extract_anomalies first.")
        if self.config.model.uses_masks:
            max_class = max(
                int(round(float(record.metadata.get("label", 0))))
                for record in records
            )
            self.config.model.parameters.set_model_param(
                "num_anomaly_classes", max_class
            )
        return self.datasets.real_anomalies(
            return_artifacts=self.config.model.parameters.input_artefacts,
            load_to_ram=True,
            dtype=torch.float32,
        )

    def train_generator(self, no_of_trials):
        self._log_step("Training generator model.")
        dataset = self._training_dataset()
        optimize(no_of_trials, self.config, dataset)
        return self.load_generator(trial_id=-1 if no_of_trials > 1 else -2)

    def load_generator(self, path_to_db_file=None, trial_id=-1):
        self._log_step("Loading generator model.")
        storage = (
            self.config.study.paths.optuna_storage_url
            if path_to_db_file is None
            else "sqlite:///" + str(path_to_db_file)
        )
        study = optuna.load_study(study_name=self.config.study.name, storage=storage)
        if trial_id == -1:
            trial = study.best_trial
        elif trial_id == -2:
            trial = max(study.get_trials(), key=lambda value: value.number)
        else:
            trial = next(
                (value for value in study.get_trials() if value.number == trial_id), None
            )
            if trial is None:
                raise ValueError(f"Optuna trial {trial_id} does not exist.")
        self._generator_model = get_model_spec(trial.user_attrs["model_name"]).build(
            trial.user_attrs["params"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator_model.to(device)
        self._generator_model.warmup(
            self.config.extraction.anomaly_size,
            device=device,
            dtype=self.config.training.dtype,
            config=self.config.training,
        )
        self._generator_model.load_checkpoint(trial.user_attrs["model_path"])
        return self._generator_model

    def train_fusion_backend(
        self,
        sample_dataloader,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        checkpoint_path: str | None = None,
        device=None,
    ):
        self._log_step("Training fusion backend.")
        spec = get_fusion_backend_spec(self.config.fusion.backend)
        if not spec.trainable:
            raise ValueError(
                f"Fusion backend {self.config.fusion.backend!r} is not trainable."
            )
        backend = self._ensure_fusion_backend()
        if spec.trainable and checkpoint_path is None:
            checkpoint_path = str(
                Path(self.config.study.paths.trained_fusion_backends)
                / f"{self.config.fusion.backend}.pth"
            )
        summary = backend.train_model(
            sample_dataloader,
            epochs=epochs,
            lr=lr,
            checkpoint_path=checkpoint_path,
            device=device,
            config=self.config.fusion,
        )
        resolved_checkpoint = (
            summary.get("checkpoint_path", checkpoint_path)
            if isinstance(summary, dict)
            else checkpoint_path
        )
        if resolved_checkpoint:
            self.config.fusion.checkpoint = resolved_checkpoint
        return summary

    def _ensure_fusion_backend(self) -> FusionBackend:
        if self._fusion_backend is not None:
            return self._fusion_backend
        self._fusion_backend = get_fusion_backend_spec(self.config.fusion.backend).build(
            {"fusion_params": self.config.fusion.parameters}
        )
        checkpoint = self.config.fusion.checkpoint
        if checkpoint:
            self._fusion_backend.load_checkpoint(checkpoint)
        return self._fusion_backend

    def generate_synthetic_anomalies(self) -> list[SyntheticAnomaly]:
        """Generate the configured number of variants for every real anomaly."""
        self._log_step("Generating synthetic anomaly variants.")
        generator_model = self._generator_model
        if generator_model is None:
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
                    self.config.augmentation,
                    anomaly_size=self.config.extraction.anomaly_size,
                    background_threshold=generation.background_threshold,
                    seed=seed,
                )
                with _seeded_random(seed):
                    image, mask = self._generate_variant(
                        generator_model,
                        sample,
                        real_image,
                        target_mask_generator,
                    )
                image, mask = _validate_generated_variant(image, mask, real_image)
                synthetic_id = stable_id("synthetic", real_id, variant_index)
                image_path = self.artifact_store.save_entity_array(
                    "synthetic_anomalies", synthetic_id, "image", image
                )
                segmentation_path = self.artifact_store.save_entity_array(
                    "synthetic_anomalies", synthetic_id, "segmentation", mask
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

    def _generate_variant(
        self,
        generator_model: GenerativeBackend,
        sample,
        real_image,
        target_mask_generator,
    ):
        generation = self.config.generation
        kwargs = {
            "mode": generation.sampling_mode,
            "variation_strength": generation.variation_strength,
            "clamp_01": generation.clamp_output,
            "target_mask_generator": target_mask_generator,
        }
        if not generation.feedback.enabled:
            image, mask = generator_model.generate(sample, **kwargs)
            return _as_numpy(image), _as_numpy(mask)

        best_score = -np.inf
        best = None
        threshold = float(generation.feedback.similarity_threshold)
        for attempt in range(int(generation.feedback.max_attempts)):
            image, mask = generator_model.generate(sample, **kwargs)
            image, mask = _as_numpy(image), _as_numpy(mask)
            if image.shape != real_image.shape:
                raise ValueError(f"Generated shape {image.shape} differs from {real_image.shape}.")
            score = ssim_01(real_image, image)
            if score > best_score:
                best_score = score
                best = image, mask
            if score >= threshold:
                break
            if (attempt + 1) % 100 == 0:
                threshold *= float(generation.feedback.threshold_relaxation_factor)
        if best is None:
            raise RuntimeError("Generator produced no variant.")
        return best

    def plan_hybrid_samples(self, control_samples) -> list[HybridSample]:
        self._log_step("Planning hybrid samples and placements.")
        planned = build_hybrid_plan(
            control_samples,
            self.repository,
            self.artifact_store,
            self.config.matching,
        )
        if not planned:
            raise ValueError("Matching produced no hybrid sample plans.")
        return planned

    def materialize_hybrid_samples(self, *, raise_on_error: bool = True) -> list[HybridSample]:
        """Fuse all planned placements and update hybrid/placement artifact records."""
        self._log_step("Materializing planned hybrid samples.")
        hybrids = self.repository.list_hybrid_samples()
        if not hybrids:
            raise ValueError("No hybrid plan found. Run plan_hybrid_samples first.")
        synthetic_dataset = self.datasets.synthetic_anomalies(
            load_to_ram=False,
            dtype=torch.float32,
            numpy_mode=True,
        )
        if not len(synthetic_dataset):
            raise ValueError("No synthetic anomalies found.")
        fusion_backend = self._ensure_fusion_backend()
        generated: list[HybridSample] = []
        failures = []

        for hybrid in hybrids:
            try:
                result = self._materialize_hybrid(
                    hybrid, synthetic_dataset, fusion_backend
                )
                generated.append(result)
            except Exception as exc:
                failed = replace(
                    hybrid,
                    image_path=None,
                    segmentation_path=None,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.repository.upsert_hybrid_sample(failed)
                failures.append(failed)

        if failures and raise_on_error:
            details = "; ".join(f"{item.id}: {item.error}" for item in failures)
            raise RuntimeError(f"Failed to materialize {len(failures)} hybrid samples: {details}")
        return generated

    def _materialize_hybrid(
        self,
        hybrid: HybridSample,
        synthetic_dataset,
        fusion_backend: FusionBackend,
    ) -> HybridSample:
        original = self.repository.get_original_sample(hybrid.original_sample_id)
        image = self.artifact_store.load_array(original.image_path).copy()
        if original.segmentation_path:
            segmentation = self.artifact_store.load_array(original.segmentation_path)
            segmentation = _mask_like_image(segmentation, image)
        else:
            segmentation = np.zeros_like(image, dtype=np.uint8)
        fusion_backend.warmup(image.shape, config=self.config.fusion)

        placements = self.repository.list_placements(hybrid.id)
        if not placements:
            raise ValueError("A hybrid sample must have at least one placement.")
        for placement in placements:
            sample = synthetic_dataset.load_sample_by_id(
                placement.synthetic_anomaly_id
            )
            with _seeded_random(
                stable_seed(self.config.study.seed, placement.id, "fusion")
            ):
                output = fusion_backend.fuse(
                    sample,
                    image,
                    placement.position,
                    extraction_config=self.config.extraction,
                )
            image = output.image
            segmentation = combine_label_masks(
                segmentation, _mask_like_image(output.segmentation, image), overwrite=True
            )
            roi_image_path = None
            roi_segmentation_path = None
            if output.roi is not None:
                roi_image_path = self.artifact_store.save_entity_array(
                    "placements", placement.id, "roi_image", output.roi
                )
            if output.roi_mask is not None:
                roi_segmentation_path = self.artifact_store.save_entity_array(
                    "placements", placement.id, "roi_segmentation", output.roi_mask
                )
            self.repository.upsert_placement(
                replace(
                    placement,
                    roi_image_path=roi_image_path,
                    roi_segmentation_path=roi_segmentation_path,
                )
            )

        image_path = self.artifact_store.save_entity_array(
            "hybrid_samples", hybrid.id, "image", image
        )
        segmentation_path = self.artifact_store.save_entity_array(
            "hybrid_samples", hybrid.id, "segmentation", segmentation
        )
        generated = replace(
            hybrid,
            image_path=image_path,
            segmentation_path=segmentation_path,
            status="generated",
            error=None,
        )
        self.repository.upsert_hybrid_sample(generated)
        return generated


def _position_columns(position):
    if len(position) == 2:
        return None, position[0], position[1]
    if len(position) == 3:
        return position[0], position[1], position[2]
    raise ValueError(f"Expected a 2D or 3D normalized position, got {position!r}.")


def _mask_like_image(mask, image):
    mask = np.asarray(mask)
    if mask.shape == image.shape:
        return mask.copy()
    if mask.ndim == image.ndim and mask.shape[1:] == image.shape[1:] and mask.shape[0] == 1:
        return np.repeat(mask, image.shape[0], axis=0)
    if mask.ndim == image.ndim - 1 and mask.shape == image.shape[1:]:
        return np.repeat(mask[None, ...], image.shape[0], axis=0)
    raise ValueError(f"Mask shape {mask.shape} is incompatible with image shape {image.shape}.")


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
            f"Generated image shape {image.shape} differs from expected {expected_shape}."
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


@contextmanager
def _seeded_random(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
