from __future__ import annotations

import os
import tempfile
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
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-", suffix=".npy", dir=target.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.save(temporary, np.asarray(array), allow_pickle=False)
        try:
            os.replace(temporary_path, target)
        except BaseException:
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
