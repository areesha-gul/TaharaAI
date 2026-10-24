"""Normalize cached native/OCR page content and write processed JSON."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from classifier import ClassifyResult, PageResult

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def _ocr_image(image_bytes: bytes) -> str:
    """Run OCR on a rendered image held in memory via a short-lived file."""
    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        image_file.write(image_bytes)
        image_file.flush()
        result = _paddle_ocr().ocr(image_file.name, cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            lines.append(line[1][0])
    return "\n".join(lines).strip()


def _normalize_hyphenation(text: str) -> str:
    return re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)


def _strip_repeated_marginal_lines(pages: list[dict]) -> None:
    if len(pages) < 2:
        return
    occurrences: dict[str, set[int]] = {}
    for index, page in enumerate(pages):
        lines = page["text"].splitlines()
        for line in set(lines[:3] + lines[-3:]):
            normalized = " ".join(line.split()).casefold()
            if normalized:
                occurrences.setdefault(normalized, set()).add(index)
    repeated = {
        line for line, page_indexes in occurrences.items()
        if len(page_indexes) / len(pages) > 0.5
    }
    if not repeated:
        return
    for page in pages:
        lines = page["text"].splitlines()
        page["text"] = "\n".join(
            line for index, line in enumerate(lines)
            if not (index < 3 or index >= len(lines) - 3)
            or " ".join(line.split()).casefold() not in repeated
        ).strip()


def _extract_page(page: PageResult) -> dict:
    text = page.text
    error = page.error
    if not error and page.kind == "scanned":
        try:
            if page.image_bytes:
                text = _ocr_image(page.image_bytes)
        except Exception as exc:
            error = str(exc)
            log.exception("OCR failed for page %d", page.page_num)
    text = _normalize_hyphenation(text)
    return {"page": page.page_num, "kind": page.kind, "text": text, **({"error": error} if error else {})}


def extract(pdf_path: Path, classify_result: ClassifyResult) -> dict:
    """Create page-level output; page failures are recorded without aborting the file."""
    pages_out = [_extract_page(page) for page in classify_result.pages]
    _strip_repeated_marginal_lines(pages_out)
    return {
        "doc_name": pdf_path.stem,
        "source": str(pdf_path.resolve()),
        "kind": classify_result.kind,
        "pages": pages_out,
    }


def extract_and_save(pdf_path: Path, classify_result: ClassifyResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = extract(pdf_path, classify_result)
    out_path = out_dir / f"{pdf_path.stem}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
