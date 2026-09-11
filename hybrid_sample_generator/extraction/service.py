"""Application service for extracting and persisting real anomalies."""

from __future__ import annotations

from hybrid_sample_generator.configuration.extraction import ExtractionConfiguration
from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from hybrid_sample_generator.domain.records import RealAnomaly
from hybrid_sample_generator.extraction.extraction_2d import (
    crop_and_center_anomaly_2d,
)
from hybrid_sample_generator.extraction.extraction_3d import (
    crop_and_center_anomaly_3d,
)
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.identifiers import stable_id
from hybrid_sample_generator.persistence.study_repository import StudyRepository


class ExtractionService:
    """Extract real anomaly components and persist their records and artifacts."""

    def __init__(
        self,
        config: ExtractionConfiguration,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        datasets: StudyDatasets,
    ) -> None:
        self.config = config
        self.repository = repository
        self.artifact_store = artifact_store
        self.datasets = datasets

    def extract(self) -> list[RealAnomaly]:
        """Extract real anomalies from persisted, annotated originals."""
        source_dataset = self.datasets.original_samples(
            return_artifacts=("img", "ori_mask", "record"),
            has_anomaly=True,
            is_annotated=True,
            load_to_ram=False,
            numpy_mode=True,
        )
        if not len(source_dataset):
            raise ValueError(
                "No anomalous originals found. Run ingest_dataset with annotated "
                "anomaly samples first."
            )

        self.repository.clear_real_anomalies_and_downstream()
        extracted: list[RealAnomaly] = []

        for sample in source_dataset:
            original = sample["record"]
            image = sample["img"]
            segmentation = sample["ori_mask"]
            if segmentation is None:
                raise RuntimeError(
                    f"Anomalous original {original.id} has no segmentation artifact."
                )

            if image.ndim == 3:
                result = crop_and_center_anomaly_2d(
                    image,
                    segmentation,
                    self.config,
                )
            elif image.ndim == 4:
                result = crop_and_center_anomaly_3d(
                    image,
                    segmentation,
                    self.config,
                )
            else:
                raise ValueError(
                    f"Unexpected shape {image.shape}; expected (C,H,W) or (C,D,H,W)."
                )

            if not result or result[0] is None:
                continue

            anomalies, anomaly_rois, masks, roi_masks = result
            if not (
                len(anomalies)
                == len(anomaly_rois)
                == len(masks)
                == len(roi_masks)
            ):
                raise RuntimeError("Extraction returned unaligned anomaly artifacts.")

            for component_index, ((anomaly, metadata), roi, mask, roi_mask) in enumerate(
                zip(anomalies, anomaly_rois, masks, roi_masks)
            ):
                record_id = stable_id("real", original.id, component_index)
                image_path = self.artifact_store.save_entity_array(
                    "real_anomalies", record_id, "image", anomaly
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
                position = tuple(
                    float(value) for value in metadata["centroid_norm"]
                )
                position_z, position_y, position_x = _position_columns(position)
                record_metadata = dict(metadata)
                record_metadata["roi_shape"] = tuple(
                    int(value) for value in roi.shape[1:]
                )
                record = RealAnomaly(
                    id=record_id,
                    original_sample_id=original.id,
                    component_index=component_index,
                    image_path=image_path,
                    segmentation_path=segmentation_path,
                    roi_image_path=roi_image_path,
                    roi_segmentation_path=roi_segmentation_path,
                    spatial_dimensions=anomaly.ndim - 1,
                    position_z=position_z,
                    position_y=position_y,
                    position_x=position_x,
                    metadata=record_metadata,
                )
                self.repository.upsert_real_anomaly(record)
                extracted.append(record)

        if not extracted:
            raise ValueError("No real anomalies were extracted from the supplied samples.")
        return extracted


def _position_columns(
    position: tuple[float, ...],
) -> tuple[float | None, float, float]:
    if len(position) == 2:
        return None, position[0], position[1]
    if len(position) == 3:
        return position[0], position[1], position[2]
    raise ValueError(
        f"Expected a 2D or 3D normalized position, got {position!r}."
    )
