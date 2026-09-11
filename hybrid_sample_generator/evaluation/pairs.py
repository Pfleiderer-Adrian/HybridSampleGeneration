"""Resolve persisted real/synthetic anomaly pairs for evaluation."""

from dataclasses import dataclass

from hybrid_sample_generator.persistence.study_repository import StudyRepository


@dataclass(frozen=True)
class EvaluationPair:
    id: str
    real_anomaly_id: str
    synthetic_anomaly_id: str
    real_image_path: str
    synthetic_image_path: str
    real_mask_path: str | None = None
    synthetic_mask_path: str | None = None
    placement_id: str | None = None


def cutout_pairs(repository: StudyRepository) -> list[EvaluationPair]:
    pairs = []
    for synthetic in repository.list_synthetic_anomalies():
        real = repository.get_real_anomaly(synthetic.real_anomaly_id)
        pairs.append(
            EvaluationPair(
                id=synthetic.id,
                real_anomaly_id=real.id,
                synthetic_anomaly_id=synthetic.id,
                real_image_path=real.image_path,
                synthetic_image_path=synthetic.image_path,
                real_mask_path=real.segmentation_path,
                synthetic_mask_path=synthetic.segmentation_path,
            )
        )
    return pairs


def roi_pairs(repository: StudyRepository) -> list[EvaluationPair]:
    pairs = []
    for entry in repository.hierarchy():
        placement = entry.placement
        if placement.roi_image_path is None:
            continue
        pairs.append(
            EvaluationPair(
                id=placement.id,
                real_anomaly_id=entry.real_anomaly.id,
                synthetic_anomaly_id=entry.synthetic_anomaly.id,
                real_image_path=entry.real_anomaly.roi_image_path,
                synthetic_image_path=placement.roi_image_path,
                real_mask_path=entry.real_anomaly.roi_segmentation_path,
                synthetic_mask_path=placement.roi_segmentation_path,
                placement_id=placement.id,
            )
        )
    return pairs


__all__ = ["EvaluationPair", "cutout_pairs", "roi_pairs"]
