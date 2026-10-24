"""Classify and read PDF pages using one PyMuPDF document open."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MIN_CHARS = int(os.getenv("MIN_NATIVE_CHARS", "50"))
FULL_PAGE_IMAGE_RATIO = float(os.getenv("FULL_PAGE_IMAGE_RATIO", "0.70"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))


@dataclass
class PageResult:
    page_num: int
    kind: str
    char_count: int
    text: str = ""
    image_bytes: bytes | None = None
    error: str | None = None


@dataclass
class ClassifyResult:
    path: Path
    kind: str
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None


def _image_coverage(page) -> float:
    page_area = max(page.rect.get_area(), 1.0)
    largest_image_area = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            clipped = rect & page.rect
            largest_image_area = max(largest_image_area, clipped.get_area())
    return largest_image_area / page_area


def classify_pdf(pdf_path: Path) -> ClassifyResult:
    """Inspect each page once; capture text or a rendered OCR image per page."""
    pages = []
    try:
        with pymupdf.open(pdf_path) as document:
            for index, page in enumerate(document):
                page_num = index + 1
                try:
                    text = page.get_text("text").strip()
                    char_count = len(text)
                    full_image = _image_coverage(page) >= FULL_PAGE_IMAGE_RATIO
                    kind = "scanned" if char_count < MIN_CHARS or full_image else "native"
                    image_bytes = None
                    if kind == "scanned":
                        image_bytes = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
                    pages.append(PageResult(
                        page_num=page_num,
                        kind=kind,
                        char_count=char_count,
                        text=text if kind == "native" else "",
                        image_bytes=image_bytes,
                    ))
                except Exception as exc:
                    log.exception("Classification failed for %s page %d", pdf_path, page_num)
                    pages.append(PageResult(
                        page_num=page_num,
                        kind="scanned",
                        char_count=0,
                        error=str(exc),
                    ))
    except Exception as exc:
        return ClassifyResult(path=pdf_path, kind="error", error=str(exc))

    kinds = {page.kind for page in pages if not page.error}
    if kinds == {"native"}:
        doc_kind = "native"
    elif kinds == {"scanned"} or not kinds:
        doc_kind = "scanned"
    else:
        doc_kind = "mixed"
    return ClassifyResult(path=pdf_path, kind=doc_kind, pages=pages)
