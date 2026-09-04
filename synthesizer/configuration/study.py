from __future__ import annotations

import os
from dataclasses import dataclass

from synthesizer.StudyPaths import StudyPaths


@dataclass
class StudyConfiguration:
    """Identity, storage location and reproducibility settings for one study."""

    name: str
    folder: str
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("study.name must not be empty.")
        self.folder = os.path.normpath(os.fspath(self.folder))
        self.seed = int(self.seed)

    @property
    def paths(self) -> StudyPaths:
        return StudyPaths(self.folder, self.name)
