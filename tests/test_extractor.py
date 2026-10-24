import pymupdf as fitz
from unittest.mock import patch

from classifier import ClassifyResult, PageResult, classify_pdf
from extractor import extract


def test_pymupdf_classification_and_repeated_header_cleanup(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    for number in range(2):
        page = document.new_page()
        page.insert_text((40, 35), "REPEATED HEADER")
        page.insert_text((40, 80), f"Page {number + 1} has enough native text for classification.")
        wrapped_word = "hyphen-\nated word" if number == 0 else "securi-\nty policy"
        page.insert_text((40, 110), wrapped_word)
    document.save(pdf_path)
    document.close()

    classification = classify_pdf(pdf_path)
    output = extract(pdf_path, classification)

    assert classification.kind == "native"
    assert len(output["pages"]) == 2
    assert all("REPEATED HEADER" not in page["text"] for page in output["pages"])
    assert "hyphenated word" in output["pages"][0]["text"]
    assert "security policy" in output["pages"][1]["text"]


def test_ocr_failure_is_recorded_and_does_not_abort_later_pages(tmp_path):
    pdf_path = tmp_path / "pages.pdf"
    result = ClassifyResult(
        path=pdf_path,
        kind="mixed",
        pages=[
            PageResult(page_num=1, kind="scanned", char_count=0, image_bytes=b"image"),
            PageResult(page_num=2, kind="native", char_count=10, text="later page"),
        ],
    )

    with patch("extractor._ocr_image", side_effect=RuntimeError("OCR fixture failure")):
        output = extract(pdf_path, result)

    assert output["pages"][0]["error"] == "OCR fixture failure"
    assert output["pages"][1]["text"] == "later page"
    assert output["pages"][1]["page"] == 2
