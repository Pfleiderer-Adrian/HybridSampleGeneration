"""Orchestrate the complete record-based generation pipeline."""

from __future__ import annotations

from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from hybrid_sample_generator.extraction.service import ExtractionService
from hybrid_sample_generator.fusion.interfaces import FusionBackend
from hybrid_sample_generator.fusion.service import FusionService
from hybrid_sample_generator.generation.interfaces import GenerativeBackend
from hybrid_sample_generator.generation.service import GenerationService
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.pipeline.ingestion import (
    DatasetSummary,
    ingest_dataset as ingest_study_dataset,
)
from hybrid_sample_generator.matching.planner import (
    plan_hybrid_samples as build_hybrid_plan,
)
from hybrid_sample_generator.domain.records import HybridSample, RealAnomaly, SyntheticAnomaly
from hybrid_sample_generator.persistence.study_repository import StudyRepository


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
        self._extraction_service = ExtractionService(
            config.extraction,
            self.repository,
            self.artifact_store,
            self.datasets,
        )
        self._generation_service = GenerationService(
            config,
            self.repository,
            self.artifact_store,
            self.datasets,
            model=generator_model,
        )
        self._fusion_service = FusionService(
            config.fusion,
            config.extraction,
            config.study.seed,
            self.repository,
            self.artifact_store,
            self.datasets,
            backend=fusion_backend,
        )

    def _log_step(self, message: str) -> None:
        print(f"[HybridDataGenerator] {message}")

    def ingest_dataset(self, sample_dataloader) -> DatasetSummary:
        """Persist and classify every original sample exactly once."""
        self._log_step("Ingesting and exploring original samples.")
        summary = ingest_study_dataset(
            sample_dataloader,
            self.repository,
            self.artifact_store,
            expected_spatial_dimensions=len(self.config.extraction.anomaly_size) - 1,
            expected_channels=int(self.config.extraction.anomaly_size[0]),
        )
        self._log_step(
            f"Ingested {summary.total_samples} originals: "
            f"{summary.anomalous_samples} anomalous, "
            f"{summary.control_samples} controls."
        )
        return summary

    def extract_anomalies(self) -> list[RealAnomaly]:
        """Extract real anomalies from the persisted anomalous originals."""
        self._log_step("Extracting real anomalies into normalized study records.")
        return self._extraction_service.extract()

    def train_generator(self, no_of_trials):
        self._log_step("Training generator model.")
        self._generation_service.train(no_of_trials)
        return self.load_generator(trial_id=-1 if no_of_trials > 1 else -2)

    def load_generator(self, path_to_db_file=None, trial_id=-1):
        self._log_step("Loading generator model.")
        return self._generation_service.load(
            path_to_db_file=path_to_db_file,
            trial_id=trial_id,
        )

    def generate_synthetic_anomalies(self) -> list[SyntheticAnomaly]:
        """Generate the configured number of variants for every real anomaly."""
        self._log_step("Generating synthetic anomaly variants.")
        return self._generation_service.generate()

    def plan_hybrid_samples(self) -> list[HybridSample]:
        self._log_step("Planning hybrid samples and placements.")
        planned = build_hybrid_plan(
            self.repository,
            self.artifact_store,
            self.config.matching,
        )
        if not planned:
            raise ValueError("Matching produced no hybrid sample plans.")
        return planned

    def materialize_hybrid_samples(
        self,
        *,
        raise_on_error: bool = True,
    ) -> list[HybridSample]:
        """Fuse all planned placements and update their artifact records."""
        self._log_step("Materializing planned hybrid samples.")
        return self._fusion_service.materialize(raise_on_error=raise_on_error)
