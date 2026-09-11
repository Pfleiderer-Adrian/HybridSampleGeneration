"""Public facade for the canonical SQLite study repository."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hybrid_sample_generator.persistence.schema import StudySchemaMixin
from hybrid_sample_generator.persistence.study_reader import StudyReaderMixin
from hybrid_sample_generator.persistence.study_writer import StudyWriterMixin


class StudyRepository(StudySchemaMixin, StudyWriterMixin, StudyReaderMixin):
    """Canonical SQLite index for all study entities and relationships."""

    def __init__(self, database_path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


__all__ = ["StudyRepository"]
