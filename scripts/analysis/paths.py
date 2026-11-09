"""Shared locations for research outputs, without creating directories on import."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("HSG_RESULTS_ROOT", "/mnt/results/mvtec2")).expanduser().resolve()
REPORTS_ROOT = RESULTS_ROOT / "reports"
AUDITS_ROOT = RESULTS_ROOT / "audits"
NOTES_ARCHIVE_ROOT = RESULTS_ROOT / "notes_archive"


def report_dir(name):
    """Return a report's data directory; callers retain control of file writes."""
    return REPORTS_ROOT / name


def audit_dir(name):
    """Return an audit's data directory; no directory is created implicitly."""
    return AUDITS_ROOT / name
