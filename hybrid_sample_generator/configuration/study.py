"""Study identity, storage, and reproducibility configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hybrid_sample_generator.configuration.parameters import param
from hybrid_sample_generator.persistence.study_paths import StudyPaths


@dataclass
class StudyConfiguration:
    """Identity, storage location and reproducibility settings for one study."""

    name: str = param(
        required=True,
        doc="Study name supplied when creating Configuration.",
    )
    folder: str = param(
        default_label="results/<study name>",
        doc="Directory for the database, models, and generated artifacts; defaults to results/<study name> in the working directory.",
    )
    seed: int = param(42, doc="Seed for reproducible random decisions.")

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("study.name must not be empty.")
        self.folder = os.path.normpath(os.fspath(self.folder))
        self.seed = int(self.seed)

    @property
    def paths(self) -> StudyPaths:
        return StudyPaths(self.folder, self.name)
