import re

import chunker


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return list(range(len(re.findall(r"\S+", text))))


def test_clause_parser_detects_real_ids_and_ignores_year_page_numbers(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    text = (
        "2023\n42\n1 Introduction\nOpening text\n"
        "1.1 Access control\nAccess rules\n"
        "A.5.15 Access rights\nReview rights\n"
        "Annex A - Mapping\nMapping content\n"
    )

    sections = chunker.parse_sections(text)

    assert [section.section_id for section in sections] == ["1", "1.1", "A.5.15", "Annex A"]
    assert all(section.section_id not in {"2023", "42"} for section in sections)


def test_clause_patterns_are_case_sensitive_and_ignore_year_page_only_lines():
    sections = chunker.parse_sections("a.5.15 lowercase\nBody\nA.5.15 Valid clause\nValid body\n2024\n8")
    assert [section.section_id for section in sections if section.section_id] == ["A.5.15"]


def test_chunks_respect_token_hard_limit_and_preserve_line_breaks(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    text = "\n".join(f"row{i} value{i}" for i in range(1100))

    chunks = chunker.chunk_text(text, page_num=3)

    assert len(chunks) > 1
    assert all(len(WordTokenizer().encode(chunk.text)) <= 480 for chunk in chunks)
    assert all(chunk.page_num == 3 for chunk in chunks)
    assert any("\n" in chunk.text for chunk in chunks)


def test_short_sibling_clauses_merge_without_losing_text(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    sections = chunker.parse_sections("1.1 Access control\nBrief rule\n1.2 Reviews\nReview rule")

    merged = chunker._merge_short_sections(sections)

    assert len(merged) == 1
    assert "Brief rule" in merged[0].text
    assert "Review rule" in merged[0].text
    assert merged[0].section_id == "1.1"
    assert merged[0].section_ids == ("1.1", "1.2")
    assert " / " in merged[0].heading_path


def test_short_sibling_clauses_merge_across_pages(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    pages = [
        {"page": 4, "text": "1.1 Access control\nShort requirement"},
        {"page": 5, "text": "1.2 Access review\nAnother short requirement"},
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].section_id == "1.1"
    assert chunks[0].section_ids == ("1.1", "1.2")
    assert chunks[0].page_num == 4
    assert chunks[0].page_nums == (4, 5)


def test_unheaded_page_continuation_keeps_previous_clause_metadata(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    chunks = chunker.chunk_pages([
        {"page": 4, "text": "1.1 Access control\nFirst part"},
        {"page": 5, "text": "Continued requirement text"},
    ])

    assert len(chunks) == 1
    assert chunks[0].section_id == "1.1"
    assert chunks[0].page_nums == (4, 5)


def test_heading_path_inherits_parent_across_pages(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    chunks = chunker.chunk_pages([
        {"page": 1, "text": "A.5 Access control\nParent context"},
        {"page": 2, "text": "A.5.15 Access rights\nChild context"},
    ])

    child = next(chunk for chunk in chunks if chunk.section_id == "A.5.15")
    assert child.heading_path == "A.5 Access control > A.5.15 Access rights"


def test_chunk_indexes_continue_from_given_start(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer", lambda: WordTokenizer())
    chunks = chunker.chunk_text(" ".join(f"word{i}" for i in range(700)), start_index=12)

    assert [chunk.chunk_index for chunk in chunks] == list(range(12, 12 + len(chunks)))