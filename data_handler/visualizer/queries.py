from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyRecords import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from synthesizer.StudyRepository import StudyRepository


@dataclass(frozen=True)
class AnomalyContext:
    original: OriginalSample
    real: RealAnomaly
    synthetic: SyntheticAnomaly | None
    variants: tuple[SyntheticAnomaly, ...]


@dataclass(frozen=True)
class PlacementContext:
    placement: Placement
    synthetic: SyntheticAnomaly
    real: RealAnomaly
    real_original: OriginalSample


@dataclass(frozen=True)
class HybridContext:
    original: OriginalSample
    hybrid: HybridSample
    placements: tuple[PlacementContext, ...]
    selected_placement: PlacementContext | None


@dataclass
class EvaluationGroup:
    pair_id: str
    real_anomaly_id: str
    synthetic_anomaly_id: str
    placement_id: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    calculators: set[str] = field(default_factory=set)
    score: float = 0.0

    @property
    def scope(self) -> str:
        return "placement" if self.placement_id else "cutout"

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.real_anomaly_id, self.synthetic_anomaly_id, self.placement_id


class StudyBrowserModel:
    """In-memory index of lightweight records; arrays stay in ArtifactStore."""

    def __init__(
        self,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        metric_csv_path: str | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store
        self.metric_csv_path = metric_csv_path
        self.refresh()

    def refresh(self) -> None:
        self.originals = self.repository.list_original_samples()
        self.reals = self.repository.list_real_anomalies()
        self.synthetics = self.repository.list_synthetic_anomalies()
        self.hybrids = self.repository.list_hybrid_samples()
        self.placements = self.repository.list_placements()

        self.original_by_id = {record.id: record for record in self.originals}
        self.real_by_id = {record.id: record for record in self.reals}
        self.synthetic_by_id = {record.id: record for record in self.synthetics}
        self.hybrid_by_id = {record.id: record for record in self.hybrids}
        self.placement_by_id = {record.id: record for record in self.placements}

        self.reals_by_original = _group(self.reals, "original_sample_id")
        self.synthetics_by_real = _group(self.synthetics, "real_anomaly_id")
        self.hybrids_by_original = _group(self.hybrids, "original_sample_id")
        self.placements_by_hybrid = _group(self.placements, "hybrid_sample_id")
        self.placements_by_synthetic = _group(self.placements, "synthetic_anomaly_id")
        self._match_candidates_by_original: dict[
            str, tuple[MatchCandidate, ...]
        ] = {}
        self.evaluations = load_evaluation_groups(self.metric_csv_path)

    def summary(self) -> dict[str, int]:
        status_counts = defaultdict(int)
        for hybrid in self.hybrids:
            status_counts[hybrid.status] += 1
        return {
            "originals": len(self.originals),
            "anomalous_originals": sum(record.has_anomaly for record in self.originals),
            "controls": sum(not record.has_anomaly for record in self.originals),
            "real_anomalies": len(self.reals),
            "synthetic_anomalies": len(self.synthetics),
            "hybrids": len(self.hybrids),
            "placements": len(self.placements),
            "planned": status_counts["planned"],
            "generated": status_counts["generated"],
            "failed": status_counts["failed"],
            "match_candidates": self.repository.count_match_candidates(),
            "evaluation_pairs": len(self.evaluations),
        }

    def anomaly_context(
        self,
        real_anomaly_id: str,
        synthetic_anomaly_id: str | None = None,
    ) -> AnomalyContext:
        real = self.real_by_id[real_anomaly_id]
        variants = tuple(self.synthetics_by_real.get(real.id, ()))
        synthetic = (
            self.synthetic_by_id.get(synthetic_anomaly_id)
            if synthetic_anomaly_id
            else (variants[0] if variants else None)
        )
        if synthetic is not None and synthetic.real_anomaly_id != real.id:
            raise ValueError("Synthetic anomaly does not belong to the selected real anomaly.")
        return AnomalyContext(
            original=self.original_by_id[real.original_sample_id],
            real=real,
            synthetic=synthetic,
            variants=variants,
        )

    def placement_context(self, placement_id: str) -> PlacementContext:
        placement = self.placement_by_id[placement_id]
        synthetic = self.synthetic_by_id[placement.synthetic_anomaly_id]
        real = self.real_by_id[synthetic.real_anomaly_id]
        return PlacementContext(
            placement=placement,
            synthetic=synthetic,
            real=real,
            real_original=self.original_by_id[real.original_sample_id],
        )

    def hybrid_context(
        self,
        hybrid_sample_id: str,
        placement_id: str | None = None,
    ) -> HybridContext:
        hybrid = self.hybrid_by_id[hybrid_sample_id]
        placements = tuple(
            self.placement_context(record.id)
            for record in self.placements_by_hybrid.get(hybrid.id, ())
        )
        selected = next(
            (
                context
                for context in placements
                if context.placement.id == placement_id
            ),
            placements[0] if placements else None,
        )
        return HybridContext(
            original=self.original_by_id[hybrid.original_sample_id],
            hybrid=hybrid,
            placements=placements,
            selected_placement=selected,
        )

    def first_anomaly_context(self) -> AnomalyContext | None:
        return self.anomaly_context(self.reals[0].id) if self.reals else None

    def first_hybrid_context(self) -> HybridContext | None:
        if not self.hybrids:
            return None
        hybrid = next(
            (record for record in self.hybrids if record.status == "generated"),
            self.hybrids[0],
        )
        return self.hybrid_context(hybrid.id)

    def match_candidates(
        self, original_sample_id: str
    ) -> tuple[MatchCandidate, ...]:
        if original_sample_id not in self.original_by_id:
            raise KeyError(original_sample_id)
        if original_sample_id not in self._match_candidates_by_original:
            self._match_candidates_by_original[original_sample_id] = tuple(
                self.repository.list_match_candidates(original_sample_id)
            )
        return self._match_candidates_by_original[original_sample_id]

    def missing_artifacts(self) -> list[tuple[str, str, str]]:
        missing = []
        for kind, record, fields in self._artifact_records():
            for field_name in fields:
                path = getattr(record, field_name)
                if path and not self.artifact_store.exists(path):
                    missing.append((kind, record.id, field_name))
        return missing

    def record_details(self, record) -> dict:
        return asdict(record)

    def _artifact_records(self):
        for record in self.originals:
            yield "OriginalSample", record, ("image_path", "segmentation_path")
        for record in self.reals:
            yield "RealAnomaly", record, (
                "image_path",
                "segmentation_path",
                "roi_image_path",
                "roi_segmentation_path",
            )
        for record in self.synthetics:
            yield "SyntheticAnomaly", record, ("image_path", "segmentation_path")
        for record in self.hybrids:
            yield "HybridSample", record, ("image_path", "segmentation_path")
        for record in self.placements:
            yield "Placement", record, ("roi_image_path", "roi_segmentation_path")


def load_evaluation_groups(csv_path: str | None) -> list[EvaluationGroup]:
    if not csv_path or not Path(csv_path).is_file():
        return []
    groups: dict[tuple[str, str, str | None], EvaluationGroup] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            real_id = (row.get("real_anomaly_id") or "").strip()
            synthetic_id = (row.get("synthetic_anomaly_id") or "").strip()
            if not real_id or not synthetic_id:
                continue
            placement_id = (row.get("placement_id") or "").strip() or None
            key = real_id, synthetic_id, placement_id
            group = groups.setdefault(
                key,
                EvaluationGroup(
                    pair_id=(row.get("pair_id") or synthetic_id).strip(),
                    real_anomaly_id=real_id,
                    synthetic_anomaly_id=synthetic_id,
                    placement_id=placement_id,
                ),
            )
            calculator = (row.get("feature_calculator") or "").strip()
            if calculator:
                group.calculators.add(calculator)
            try:
                values = json.loads(row.get("metric_diffs") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            for name, value in values.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    group.metrics[str(name)] = value
    return sorted(
        groups.values(),
        key=lambda group: (
            group.scope,
            group.real_anomaly_id,
            group.synthetic_anomaly_id,
            group.placement_id or "",
        ),
    )


def filter_evaluation_groups(
    groups: list[EvaluationGroup],
    *,
    metrics: tuple[str, ...] = (),
    top_percent: float = 100.0,
    scope: str = "all",
    query: str = "",
) -> list[EvaluationGroup]:
    query = query.strip().lower()
    candidates = [
        group
        for group in groups
        if (scope == "all" or group.scope == scope)
        and (
            not query
            or query in group.real_anomaly_id.lower()
            or query in group.synthetic_anomaly_id.lower()
            or query in (group.placement_id or "").lower()
        )
    ]
    metrics = tuple(metric for metric in metrics if metric)
    if not metrics:
        return [replace(group, score=0.0) for group in candidates]

    top_percent = min(max(float(top_percent), 0.1), 100.0)
    ranges = {}
    thresholds = {}
    for metric in metrics:
        values = np.asarray(
            [group.metrics[metric] for group in candidates if metric in group.metrics],
            dtype=float,
        )
        if not values.size:
            return []
        ranges[metric] = float(values.min()), float(values.max())
        thresholds[metric] = float(np.percentile(values, 100.0 - top_percent))

    filtered = []
    for group in candidates:
        if not all(
            metric in group.metrics and group.metrics[metric] >= thresholds[metric]
            for metric in metrics
        ):
            continue
        normalized = []
        for metric in metrics:
            low, high = ranges[metric]
            normalized.append(
                1.0
                if high <= low
                else (group.metrics[metric] - low) / (high - low)
            )
        filtered.append(replace(group, score=float(np.mean(normalized))))
    return sorted(filtered, key=lambda group: (-group.score, group.key))


def _group(records, field_name: str):
    grouped = defaultdict(list)
    for record in records:
        grouped[getattr(record, field_name)].append(record)
    return dict(grouped)
