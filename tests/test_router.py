from pathlib import Path

from classifier import ClassifyResult
from router import route


def test_route_keeps_original_pdf_without_copy(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    root = tmp_path / "documents"

    result = route(ClassifyResult(path=source, kind="native"), root)

    assert result == source
    assert list(tmp_path.rglob("*.pdf")) == [source]