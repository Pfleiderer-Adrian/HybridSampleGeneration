from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_handler.visualizer.queries import StudyBrowserModel


ENTITY_KINDS = {"original", "real", "synthetic", "hybrid", "placement"}


@dataclass(frozen=True)
class RemovalImpact:
    kind: str
    entity_id: str
    original_ids: tuple[str, ...]
    real_ids: tuple[str, ...]
    synthetic_ids: tuple[str, ...]
    hybrid_ids: tuple[str, ...]
    placement_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    match_candidate_count: int

    @property
    def record_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.original_ids,
                self.real_ids,
                self.synthetic_ids,
                self.hybrid_ids,
                self.placement_ids,
            )
        ) + self.match_candidate_count

    def describe(self) -> str:
        return "\n".join(
            (
                f"Original samples: {len(self.original_ids)}",
                f"Real anomalies: {len(self.real_ids)}",
                f"Synthetic anomalies: {len(self.synthetic_ids)}",
                f"Hybrid samples: {len(self.hybrid_ids)}",
                f"Placements: {len(self.placement_ids)}",
                f"Match candidates: {self.match_candidate_count}",
                f"Artifact files: {len(self.artifact_paths)}",
                "Evaluation CSV rows are not rewritten automatically.",
            )
        )


class StudyMaintenance:
    """Dependency-aware removal with recoverable artifact archival."""

    def __init__(self, model: StudyBrowserModel) -> None:
        self.model = model

    def preview_removal(self, kind: str, entity_id: str) -> RemovalImpact:
        kind = kind.strip().lower()
        if kind not in ENTITY_KINDS:
            raise ValueError(f"Unknown entity kind {kind!r}.")

        original_ids: set[str] = set()
        real_ids: set[str] = set()
        synthetic_ids: set[str] = set()
        hybrid_ids: set[str] = set()
        placement_ids: set[str] = set()

        def add_placement(placement_id: str) -> None:
            if placement_id not in self.model.placement_by_id:
                raise KeyError(placement_id)
            if placement_id in placement_ids:
                return
            placement_ids.add(placement_id)
            add_hybrid(self.model.placement_by_id[placement_id].hybrid_sample_id)

        def add_hybrid(hybrid_id: str) -> None:
            if hybrid_id not in self.model.hybrid_by_id:
                raise KeyError(hybrid_id)
            if hybrid_id in hybrid_ids:
                return
            hybrid_ids.add(hybrid_id)
            for placement in self.model.placements_by_hybrid.get(hybrid_id, ()):
                add_placement(placement.id)

        def add_synthetic(synthetic_id: str) -> None:
            if synthetic_id not in self.model.synthetic_by_id:
                raise KeyError(synthetic_id)
            synthetic_ids.add(synthetic_id)
            for placement in self.model.placements_by_synthetic.get(synthetic_id, ()):
                add_placement(placement.id)

        def add_real(real_id: str) -> None:
            if real_id not in self.model.real_by_id:
                raise KeyError(real_id)
            real_ids.add(real_id)
            for synthetic in self.model.synthetics_by_real.get(real_id, ()):
                add_synthetic(synthetic.id)

        def add_original(original_id: str) -> None:
            if original_id not in self.model.original_by_id:
                raise KeyError(original_id)
            original_ids.add(original_id)
            for real in self.model.reals_by_original.get(original_id, ()):
                add_real(real.id)
            for hybrid in self.model.hybrids_by_original.get(original_id, ()):
                add_hybrid(hybrid.id)

        {
            "original": add_original,
            "real": add_real,
            "synthetic": add_synthetic,
            "hybrid": add_hybrid,
            "placement": add_placement,
        }[kind](entity_id)

        artifact_paths: set[str] = set()
        for original_id in original_ids:
            record = self.model.original_by_id[original_id]
            artifact_paths.update(_present(record.image_path, record.segmentation_path))
        for real_id in real_ids:
            record = self.model.real_by_id[real_id]
            artifact_paths.update(
                _present(
                    record.image_path,
                    record.segmentation_path,
                    record.roi_image_path,
                    record.roi_segmentation_path,
                )
            )
        for synthetic_id in synthetic_ids:
            record = self.model.synthetic_by_id[synthetic_id]
            artifact_paths.update(_present(record.image_path, record.segmentation_path))
        for hybrid_id in hybrid_ids:
            record = self.model.hybrid_by_id[hybrid_id]
            artifact_paths.update(_present(record.image_path, record.segmentation_path))
        for placement_id in placement_ids:
            record = self.model.placement_by_id[placement_id]
            artifact_paths.update(
                _present(record.roi_image_path, record.roi_segmentation_path)
            )

        match_candidate_count = self._match_candidate_count(
            original_ids=original_ids,
            real_ids=real_ids,
        )
        return RemovalImpact(
            kind=kind,
            entity_id=entity_id,
            original_ids=tuple(sorted(original_ids)),
            real_ids=tuple(sorted(real_ids)),
            synthetic_ids=tuple(sorted(synthetic_ids)),
            hybrid_ids=tuple(sorted(hybrid_ids)),
            placement_ids=tuple(sorted(placement_ids)),
            artifact_paths=tuple(sorted(artifact_paths)),
            match_candidate_count=match_candidate_count,
        )

    def archive_and_remove(self, impact: RemovalImpact) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        trash_root = (
            self.model.artifact_store.study_folder
            / ".trash"
            / f"{timestamp}-{impact.kind}-{impact.entity_id}"
        )
        moved: list[tuple[Path, Path]] = []
        try:
            with self.model.repository.connection() as connection:
                for relative_path in impact.artifact_paths:
                    source = self.model.artifact_store.resolve(relative_path)
                    if not source.is_file():
                        continue
                    destination = trash_root / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    moved.append((source, destination))

                _delete_ids(connection, "placements", impact.placement_ids)
                _delete_ids(connection, "hybrid_samples", impact.hybrid_ids)
                _delete_ids(connection, "synthetic_anomalies", impact.synthetic_ids)
                if impact.real_ids:
                    _delete_where_ids(
                        connection,
                        "match_candidates",
                        "real_anomaly_id",
                        impact.real_ids,
                    )
                if impact.original_ids:
                    _delete_where_ids(
                        connection,
                        "match_candidates",
                        "original_sample_id",
                        impact.original_ids,
                    )
                _delete_ids(connection, "real_anomalies", impact.real_ids)
                _delete_ids(connection, "original_samples", impact.original_ids)
        except BaseException:
            for source, destination in reversed(moved):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
            raise
        self.model.refresh()
        return trash_root

    def _match_candidate_count(
        self,
        *,
        original_ids: set[str],
        real_ids: set[str],
    ) -> int:
        clauses = []
        parameters = []
        if original_ids:
            placeholders = ",".join("?" for _ in original_ids)
            clauses.append(f"original_sample_id IN ({placeholders})")
            parameters.extend(sorted(original_ids))
        if real_ids:
            placeholders = ",".join("?" for _ in real_ids)
            clauses.append(f"real_anomaly_id IN ({placeholders})")
            parameters.extend(sorted(real_ids))
        if not clauses:
            return 0
        with self.model.repository.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM match_candidates WHERE " + " OR ".join(clauses),
                tuple(parameters),
            ).fetchone()
        return int(row[0])


def _delete_ids(connection, table: str, ids: tuple[str, ...]) -> None:
    _delete_where_ids(connection, table, "id", ids)


def _delete_where_ids(
    connection,
    table: str,
    column: str,
    ids,
) -> None:
    ids = tuple(ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    connection.execute(
        f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
        ids,
    )


def _present(*paths: str | None) -> set[str]:
    return {path for path in paths if path}
