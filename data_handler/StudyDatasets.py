from __future__ import annotations

from data_handler.AnomalyDataset import (
    HybridSampleDataset,
    OriginalSampleDataset,
    RealAnomalyDataset,
    SyntheticAnomalyDataset,
)
from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyRepository import StudyRepository


class StudyDatasets:
    """Creates short-lived dataset views without retaining array snapshots."""

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store

    def real_anomalies(self, **kwargs) -> RealAnomalyDataset:
        return RealAnomalyDataset(self.repository, self.artifact_store, **kwargs)

    def original_samples(self, **kwargs) -> OriginalSampleDataset:
        return OriginalSampleDataset(self.repository, self.artifact_store, **kwargs)

    def synthetic_anomalies(self, **kwargs) -> SyntheticAnomalyDataset:
        return SyntheticAnomalyDataset(self.repository, self.artifact_store, **kwargs)

    def hybrid_samples(self, **kwargs) -> HybridSampleDataset:
        return HybridSampleDataset(self.repository, self.artifact_store, **kwargs)
