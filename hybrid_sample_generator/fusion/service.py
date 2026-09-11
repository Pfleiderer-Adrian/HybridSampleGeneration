"""Application service for materializing planned hybrid samples."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from tqdm import tqdm

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from hybrid_sample_generator.domain.records import HybridSample
from hybrid_sample_generator.fusion.interfaces import FusionBackend
from hybrid_sample_generator.fusion.registry import get_fusion_backend_spec
from hybrid_sample_generator.fusion.settings import FusionSettings
from hybrid_sample_generator.imaging.masks.composition import combine_label_masks
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.identifiers import stable_seed
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.randomness import seeded_random


class FusionService:
    """Own the fusion backend and materialize persisted hybrid plans."""

    def __init__(
        self,
        fusion_config: FusionSettings,
        extraction_config: ExtractionConfiguration,
        study_seed: int,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        datasets: StudyDatasets,
        *,
        backend: FusionBackend | None = None,
    ) -> None:
        self.fusion_config = fusion_config
        self.extraction_config = extraction_config
        self.study_seed = int(study_seed)
        self.repository = repository
        self.artifact_store = artifact_store
        self.datasets = datasets
        self._backend = backend
        self._backend_injected = backend is not None
        self._backend_settings = None

    def materialize(
        self,
        *,
        raise_on_error: bool = True,
    ) -> list[HybridSample]:
        """Fuse all planned placements and persist their generated artifacts."""
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

        backend = self._ensure_backend()
        generated: list[HybridSample] = []
        failures: list[HybridSample] = []

        for hybrid in tqdm(
            hybrids,
            desc="Materializing hybrid samples",
            unit="sample",
        ):
            try:
                generated.append(
                    self._materialize_hybrid(
                        hybrid,
                        synthetic_dataset,
                        backend,
                    )
                )
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
            raise RuntimeError(
                f"Failed to materialize {len(failures)} hybrid samples: {details}"
            )
        return generated

    def _ensure_backend(self) -> FusionBackend:
        if self._backend_injected:
            return self._backend

        settings = self.fusion_config.to_dict()
        if self._backend is not None and settings == self._backend_settings:
            return self._backend

        backend = get_fusion_backend_spec(self.fusion_config.backend).build(
            self.fusion_config.parameters
        )
        self._backend = backend
        self._backend_settings = settings
        return backend

    def _materialize_hybrid(
        self,
        hybrid: HybridSample,
        synthetic_dataset,
        backend: FusionBackend,
    ) -> HybridSample:
        original = self.repository.get_original_sample(hybrid.original_sample_id)
        image = self.artifact_store.load_array(original.image_path).copy()
        if original.segmentation_path:
            segmentation = self.artifact_store.load_array(
                original.segmentation_path
            )
            segmentation = _mask_like_image(segmentation, image)
        else:
            segmentation = np.zeros_like(image, dtype=np.uint8)

        backend.warmup(image.shape, config=self.fusion_config)
        placements = self.repository.list_placements(hybrid.id)
        if not placements:
            raise ValueError("A hybrid sample must have at least one placement.")

        for placement in placements:
            sample = synthetic_dataset.load_sample_by_id(
                placement.synthetic_anomaly_id
            )
            with seeded_random(
                stable_seed(self.study_seed, placement.id, "fusion")
            ):
                output = backend.fuse(
                    sample,
                    image,
                    placement.position,
                    extraction_config=self.extraction_config,
                )

            image = output.image
            segmentation = combine_label_masks(
                segmentation,
                _mask_like_image(output.segmentation, image),
                overwrite=True,
            )
            roi_image_path = None
            roi_segmentation_path = None
            if output.roi is not None:
                roi_image_path = self.artifact_store.save_entity_array(
                    "placements",
                    placement.id,
                    "roi_image",
                    output.roi,
                )
            if output.roi_mask is not None:
                roi_segmentation_path = self.artifact_store.save_entity_array(
                    "placements",
                    placement.id,
                    "roi_segmentation",
                    output.roi_mask,
                )
            self.repository.upsert_placement(
                replace(
                    placement,
                    roi_image_path=roi_image_path,
                    roi_segmentation_path=roi_segmentation_path,
                )
            )

        image_path = self.artifact_store.save_entity_array(
            "hybrid_samples",
            hybrid.id,
            "image",
            image,
        )
        segmentation_path = self.artifact_store.save_entity_array(
            "hybrid_samples",
            hybrid.id,
            "segmentation",
            segmentation,
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


def _mask_like_image(mask, image):
    mask = np.asarray(mask)
    if mask.shape == image.shape:
        return mask.copy()
    if (
        mask.ndim == image.ndim
        and mask.shape[1:] == image.shape[1:]
        and mask.shape[0] == 1
    ):
        return np.repeat(mask, image.shape[0], axis=0)
    if mask.ndim == image.ndim - 1 and mask.shape == image.shape[1:]:
        return np.repeat(mask[None, ...], image.shape[0], axis=0)
    raise ValueError(
        f"Mask shape {mask.shape} is incompatible with image shape {image.shape}."
    )
