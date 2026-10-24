"""Record classification without duplicating source PDFs."""

from pathlib import Path

from classifier import ClassifyResult


def route(result: ClassifyResult, docs_root: Path) -> Path:
    """Return the original path; the classification is stored in processed JSON."""
    return result.path
