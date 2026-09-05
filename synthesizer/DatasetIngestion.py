from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.InputSample import iter_input_samples
from synthesizer.StudyRecords import OriginalSample
from synthesizer.StudyRepository import StudyRepository, stable_id


@dataclass(frozen=True)
class DatasetSummary:
    total_samples: int
    anomalous_samples: int
    control_samples: int
    annotated_samples: int
    unannotated_samples: int
    spatial_dimensions: int
    image_shapes: tuple[tuple[int, ...], ...]


def ingest_dataset(
    sample_dataloader,
    repository: StudyRepository,
    artifact_store: ArtifactStore,
    *,
    expected_spatial_dimensions: int,
    expected_channels: int,
) -> DatasetSummary:
    """Validate and persist the complete original dataset for one study."""
    records: list[OriginalSample] = []
    seen_ids: set[str] = set()
    seen_source_names: set[str] = set()
    image_shapes: set[tuple[int, ...]] = set()
    anomalous_samples = 0
    annotated_samples = 0

    for source_index, sample in enumerate(iter_input_samples(sample_dataloader)):
        image = np.asarray(sample.image)
        segmentation = (
            None if sample.segmentation is None else np.asarray(sample.segmentation)
        )
        _validate_sample(
            image,
            segmentation,
            expected_spatial_dimensions=expected_spatial_dimensions,
            expected_channels=expected_channels,
        )

        source_name = str(sample.source_name)
        if source_name in seen_source_names:
            raise ValueError(
                f"Original source_name {source_name!r} occurs more than once."
            )
        seen_source_names.add(source_name)

        source_identity = (
            str(Path(sample.source_image_path).expanduser().resolve())
            if sample.source_image_path
            else source_name
        )
        record_id = stable_id("original", source_identity)
        if record_id in seen_ids:
            raise ValueError(
                f"Original source identity {source_identity!r} occurs more than once."
            )
        seen_ids.add(record_id)

        is_annotated = segmentation is not None
        has_anomaly = bool(is_annotated and np.any(segmentation > 0))
        annotated_samples += int(is_annotated)
        anomalous_samples += int(has_anomaly)
        image_shapes.add(tuple(int(value) for value in image.shape))

        image_path = artifact_store.save_entity_array(
            "original_samples", record_id, "image", image
        )
        segmentation_path = None
        if segmentation is not None:
            segmentation_path = artifact_store.save_entity_array(
                "original_samples", record_id, "segmentation", segmentation
            )

        metadata = dict(sample.metadata)
        if sample.source_image_path:
            metadata["source_image_path"] = str(sample.source_image_path)
        if sample.source_segmentation_path:
            metadata["source_segmentation_path"] = str(
                sample.source_segmentation_path
            )

        records.append(
            OriginalSample(
                id=record_id,
                source_name=source_name,
                image_path=image_path,
                segmentation_path=segmentation_path,
                spatial_dimensions=image.ndim - 1,
                has_anomaly=has_anomaly,
                is_annotated=is_annotated,
                source_index=source_index,
                metadata=metadata,
            )
        )

    if not records:
        raise ValueError("The supplied dataset contains no usable original samples.")

    repository.replace_original_samples(records)
    total_samples = len(records)
    return DatasetSummary(
        total_samples=total_samples,
        anomalous_samples=anomalous_samples,
        control_samples=total_samples - anomalous_samples,
        annotated_samples=annotated_samples,
        unannotated_samples=total_samples - annotated_samples,
        spatial_dimensions=expected_spatial_dimensions,
        image_shapes=tuple(sorted(image_shapes)),
    )


def _validate_sample(
    image: np.ndarray,
    segmentation: np.ndarray | None,
    *,
    expected_spatial_dimensions: int,
    expected_channels: int,
) -> None:
    expected_ndim = expected_spatial_dimensions + 1
    if image.ndim != expected_ndim:
        raise ValueError(
            f"Expected a {expected_spatial_dimensions}D channel-first image, "
            f"got shape {image.shape}."
        )
    if image.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} image channels, got shape {image.shape}."
        )
    if segmentation is None:
        return
    if (
        segmentation.ndim != image.ndim
        or segmentation.shape[1:] != image.shape[1:]
        or segmentation.shape[0] not in (1, image.shape[0])
    ):
        raise ValueError(
            "Image and segmentation must share their spatial shape, and the "
            f"segmentation needs one or {image.shape[0]} channels: "
            f"{image.shape} vs {segmentation.shape}."
        )
