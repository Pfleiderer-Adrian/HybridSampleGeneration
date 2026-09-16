"""Optional PNG export of already materialized hybrids."""

from pathlib import Path
import numpy as np

from examples.common.image_io import save_image
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository


def export_hybrids(config) -> int:
    paths = config.study.paths
    if not Path(paths.artifact_database).is_file():
        raise ValueError("No materialized hybrids to export.")
    repository = StudyRepository(paths.artifact_database)
    store = ArtifactStore(config.study.folder)
    hybrids = repository.list_hybrid_samples(status="generated")
    if not hybrids:
        raise ValueError("No materialized hybrids to export.")
    images, masks = Path(paths.generated_images), Path(paths.generated_segmentations)
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for hybrid in hybrids:
        if not hybrid.image_path or not hybrid.segmentation_path:
            raise ValueError(f"Incomplete materialized hybrid {hybrid.id}.")
        original = repository.get_original_sample(hybrid.original_sample_id)
        name = f"{Path(original.source_name).stem}__hybrid_{hybrid.variant_index}_{hybrid.id}.png"
        save_image(store.load_array(hybrid.image_path), images / name)
        save_image(segmentation_for_png(store.load_array(hybrid.segmentation_path)), masks / name)
    return len(hybrids)


def segmentation_for_png(seg: np.ndarray) -> np.ndarray:
    mask = np.asarray(seg)
    if mask.ndim != 3:
        raise ValueError(f"Expected segmentation with shape (C,H,W), got {mask.shape}")
    return np.where(mask[:1] > 0, 255, 0).astype(np.uint8)
