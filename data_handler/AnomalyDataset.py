from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyRecords import (
    HybridSample,
    OriginalSample,
    RealAnomaly,
    SyntheticAnomaly,
)
from synthesizer.StudyRepository import StudyRepository


def save_numpy_as_npy(
    array: np.ndarray,
    out_path,
    *,
    overwrite: bool = False,
    create_dirs: bool = True,
) -> str:
    """Small standalone helper for user-facing exports outside the artifact store."""
    target = Path(out_path).expanduser().resolve()
    if target.suffix.lower() != ".npy":
        target = target.with_suffix(".npy")
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"File exists: {target}")
    np.save(target, np.asarray(array), allow_pickle=False)
    return str(target)


class _RecordDataset(Dataset):
    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        return_artifacts: Sequence[str],
        dtype: torch.dtype = torch.float32,
        transform: Callable | None = None,
        load_to_ram: bool = False,
        numpy_mode: bool = False,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store
        self.return_artifacts = tuple(return_artifacts)
        self.dtype = dtype
        self.transform = transform
        self.load_to_ram = bool(load_to_ram)
        self.numpy_mode = bool(numpy_mode)
        self.records = self._load_records()
        self._by_id = {record.id: index for index, record in enumerate(self.records)}
        self._array_cache: dict[str, np.ndarray] = {}
        if self.load_to_ram:
            for record in self.records:
                for path in self._array_paths(record):
                    self._array_cache[path] = self.artifact_store.load_array(path)

    def _load_records(self):
        raise NotImplementedError

    def _array_paths(self, record) -> tuple[str, ...]:
        raise NotImplementedError

    def _sample(self, record) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        full_sample = self._sample(self.records[index])
        sample = {name: full_sample[name] for name in self.return_artifacts}
        if "img" in sample and self.transform is not None:
            sample["img"] = self.transform(sample["img"])
        return sample

    def load_sample_by_id(self, record_id: str) -> dict:
        try:
            return self[self._by_id[record_id]]
        except KeyError as exc:
            raise KeyError(f"Record not found in dataset: {record_id}") from exc

    def _array(self, path: str):
        value = self._array_cache.get(path)
        if value is None:
            value = self.artifact_store.load_array(path)
        if self.numpy_mode:
            return value
        contiguous = np.asarray(value, dtype=_numpy_dtype(self.dtype), order="C")
        tensor = torch.from_numpy(contiguous)
        return tensor.to(self.dtype) if tensor.dtype != self.dtype else tensor


class OriginalSampleDataset(_RecordDataset):
    ALLOWED_ARTIFACTS = {
        "img",
        "ori_mask",
        "fname",
        "original_sample_id",
        "has_anomaly",
        "is_annotated",
        "metadata",
        "record",
    }

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        return_artifacts: Sequence[str] = (
            "img",
            "ori_mask",
            "fname",
            "original_sample_id",
        ),
        has_anomaly: bool | None = None,
        is_annotated: bool | None = None,
        **kwargs,
    ) -> None:
        unknown = set(return_artifacts) - self.ALLOWED_ARTIFACTS
        if unknown:
            raise ValueError(f"Unknown original sample artifacts: {sorted(unknown)}")
        self.has_anomaly = has_anomaly
        self.is_annotated = is_annotated
        super().__init__(
            repository, artifact_store, return_artifacts=return_artifacts, **kwargs
        )

    def _load_records(self) -> list[OriginalSample]:
        return self.repository.list_original_samples(
            has_anomaly=self.has_anomaly,
            is_annotated=self.is_annotated,
        )

    def _array_paths(self, record: OriginalSample) -> tuple[str, ...]:
        paths = []
        if "img" in self.return_artifacts:
            paths.append(record.image_path)
        if "ori_mask" in self.return_artifacts and record.segmentation_path is not None:
            paths.append(record.segmentation_path)
        return tuple(paths)

    def _sample(self, record: OriginalSample) -> dict:
        sample = {
            "fname": record.source_name,
            "original_sample_id": record.id,
            "has_anomaly": record.has_anomaly,
            "is_annotated": record.is_annotated,
            "metadata": dict(record.metadata),
            "record": record,
        }
        if "img" in self.return_artifacts:
            sample["img"] = self._array(record.image_path)
        if "ori_mask" in self.return_artifacts:
            sample["ori_mask"] = (
                None
                if record.segmentation_path is None
                else self._array(record.segmentation_path)
            )
        return sample


class RealAnomalyDataset(_RecordDataset):
    ALLOWED_ARTIFACTS = {
        "img", "fname", "ori_mask", "anomaly_roi", "anomaly_roi_mask",
        "anomaly_meta", "real_anomaly_id", "record",
    }

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        return_artifacts: Sequence[str] = ("img", "fname"),
        **kwargs,
    ) -> None:
        unknown = set(return_artifacts) - self.ALLOWED_ARTIFACTS
        if unknown:
            raise ValueError(f"Unknown real anomaly artifacts: {sorted(unknown)}")
        super().__init__(
            repository, artifact_store, return_artifacts=return_artifacts, **kwargs
        )

    def _load_records(self) -> list[RealAnomaly]:
        return self.repository.list_real_anomalies()

    def _array_paths(self, record: RealAnomaly) -> tuple[str, ...]:
        mapping = {
            "img": record.image_path,
            "ori_mask": record.segmentation_path,
            "anomaly_roi": record.roi_image_path,
            "anomaly_roi_mask": record.roi_segmentation_path,
        }
        return tuple(mapping[name] for name in self.return_artifacts if name in mapping)

    def _sample(self, record: RealAnomaly) -> dict:
        return {
            "img": self._array(record.image_path),
            "fname": record.id,
            "ori_mask": self._array(record.segmentation_path),
            "anomaly_roi": self._array(record.roi_image_path),
            "anomaly_roi_mask": self._array(record.roi_segmentation_path),
            "anomaly_meta": dict(record.metadata),
            "real_anomaly_id": record.id,
            "record": record,
        }


class SyntheticAnomalyDataset(_RecordDataset):
    ALLOWED_ARTIFACTS = {
        "synth_anomaly", "tgt_mask", "anomaly_roi", "anomaly_roi_mask",
        "anomaly_meta", "fname", "synthetic_anomaly_id", "real_anomaly_id", "record",
    }
    DEFAULT_ARTIFACTS = (
        "synth_anomaly", "tgt_mask", "anomaly_roi", "anomaly_roi_mask",
        "anomaly_meta", "fname", "synthetic_anomaly_id", "real_anomaly_id",
    )

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        return_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
        **kwargs,
    ) -> None:
        unknown = set(return_artifacts) - self.ALLOWED_ARTIFACTS
        if unknown:
            raise ValueError(f"Unknown synthetic anomaly artifacts: {sorted(unknown)}")
        self._real_by_id: dict[str, RealAnomaly] = {}
        super().__init__(
            repository, artifact_store, return_artifacts=return_artifacts, **kwargs
        )

    def _load_records(self) -> list[SyntheticAnomaly]:
        records = self.repository.list_synthetic_anomalies()
        real_ids = {record.real_anomaly_id for record in records}
        self._real_by_id = {
            record_id: self.repository.get_real_anomaly(record_id) for record_id in real_ids
        }
        return records

    def _array_paths(self, record: SyntheticAnomaly) -> tuple[str, ...]:
        real = self._real_by_id[record.real_anomaly_id]
        mapping = {
            "synth_anomaly": record.image_path,
            "tgt_mask": record.segmentation_path,
            "anomaly_roi": real.roi_image_path,
            "anomaly_roi_mask": real.roi_segmentation_path,
        }
        return tuple(mapping[name] for name in self.return_artifacts if name in mapping)

    def _sample(self, record: SyntheticAnomaly) -> dict:
        real = self._real_by_id[record.real_anomaly_id]
        return {
            "synth_anomaly": self._array(record.image_path),
            "tgt_mask": self._array(record.segmentation_path),
            "anomaly_roi": self._array(real.roi_image_path),
            "anomaly_roi_mask": self._array(real.roi_segmentation_path),
            "anomaly_meta": dict(real.metadata),
            "fname": record.id,
            "synthetic_anomaly_id": record.id,
            "real_anomaly_id": real.id,
            "record": record,
        }


class HybridSampleDataset(_RecordDataset):
    ALLOWED_ARTIFACTS = {
        "img", "segmentation", "fname", "hybrid_sample_id", "record",
    }

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        return_artifacts: Sequence[str] = ("img", "segmentation", "hybrid_sample_id"),
        status: str | None = "generated",
        **kwargs,
    ) -> None:
        unknown = set(return_artifacts) - self.ALLOWED_ARTIFACTS
        if unknown:
            raise ValueError(f"Unknown hybrid sample artifacts: {sorted(unknown)}")
        self.status = status
        super().__init__(
            repository, artifact_store, return_artifacts=return_artifacts, **kwargs
        )

    def _load_records(self) -> list[HybridSample]:
        return self.repository.list_hybrid_samples(status=self.status)

    def _array_paths(self, record: HybridSample) -> tuple[str, ...]:
        mapping = {"img": record.image_path, "segmentation": record.segmentation_path}
        return tuple(
            mapping[name] for name in self.return_artifacts
            if name in mapping and mapping[name] is not None
        )

    def _sample(self, record: HybridSample) -> dict:
        if record.image_path is None or record.segmentation_path is None:
            raise ValueError(f"Hybrid sample {record.id} has not been materialized.")
        return {
            "img": self._array(record.image_path),
            "segmentation": self._array(record.segmentation_path),
            "fname": record.id,
            "hybrid_sample_id": record.id,
            "record": record,
        }


def _numpy_dtype(dtype: torch.dtype):
    return {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }.get(dtype, np.float32)
