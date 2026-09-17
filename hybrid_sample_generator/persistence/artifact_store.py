"""Filesystem storage and loading of study array artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class ArtifactStore:
    """Stores NumPy payloads while the database owns identity and relations."""

    ENTITY_TYPES = {
        "original_samples",
        "real_anomalies",
        "synthetic_anomalies",
        "hybrid_samples",
        "placements",
    }

    def __init__(self, study_folder) -> None:
        self.study_folder = Path(study_folder).expanduser().resolve()
        self.root = self.study_folder / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_backups = None

    @contextmanager
    def transaction(self):
        """Restore all affected arrays if a multi-file operation fails.

        Backups live on disk rather than in RAM. Keep database writes inside
        this context so database failures also restore the previous arrays.
        This provides exception rollback, not recovery after a process crash.
        """
        if self._write_backups is not None:
            raise RuntimeError("Nested artifact transactions are not supported.")
        with tempfile.TemporaryDirectory(
            prefix=".artifact-backup-", dir=self.study_folder
        ) as backup_folder:
            self._write_backups = (Path(backup_folder), {})
            try:
                yield
            except BaseException:
                for target, backup in reversed(list(self._write_backups[1].items())):
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                raise
            finally:
                self._write_backups = None

    def relative_path(self, entity_type: str, entity_id: str, role: str) -> str:
        if entity_type not in self.ENTITY_TYPES:
            raise ValueError(f"Unknown artifact entity type: {entity_type!r}")
        if not entity_id or any(part in entity_id for part in ("/", "\\", "..")):
            raise ValueError(f"Unsafe artifact id: {entity_id!r}")
        if not role or any(part in role for part in ("/", "\\", "..")):
            raise ValueError(f"Unsafe artifact role: {role!r}")
        return (Path("artifacts") / entity_type / entity_id / f"{role}.npy").as_posix()

    def resolve(self, relative_path: str) -> Path:
        path = (self.study_folder / relative_path).resolve()
        try:
            path.relative_to(self.study_folder)
        except ValueError as exc:
            raise ValueError(f"Artifact path is outside the study folder: {relative_path}") from exc
        return path

    def save_array(self, relative_path: str, array: np.ndarray) -> str:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._write_backups is not None:
            backup_folder, backups = self._write_backups
            if target not in backups:
                backup = backup_folder / str(len(backups)) if target.exists() else None
                if backup is not None:
                    shutil.copy2(target, backup)
                backups[target] = backup
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.stem}-", suffix=".npy", dir=target.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, np.asarray(array), allow_pickle=False)
            os.replace(temporary_path, target)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return relative_path

    def save_entity_array(
        self, entity_type: str, entity_id: str, role: str, array: np.ndarray
    ) -> str:
        relative_path = self.relative_path(entity_type, entity_id, role)
        return self.save_array(relative_path, array)

    def load_array(self, relative_path: str, *, mmap_mode=None) -> np.ndarray:
        return np.load(self.resolve(relative_path), allow_pickle=False, mmap_mode=mmap_mode)

    def exists(self, relative_path: str | None) -> bool:
        return bool(relative_path) and self.resolve(relative_path).is_file()
