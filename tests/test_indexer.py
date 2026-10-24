import json

from indexer import _document_metadata, _page_chunks


def test_known_filename_metadata_fallback():
    from pathlib import Path

    metadata = _document_metadata(Path("ISO_IEC-270012022-ed.3.json"), None)

    assert metadata["standard"] == "ISO/IEC 27001"
    assert metadata["edition"] == "2022"


def test_chunk_indexes_continue_across_pages(tmp_path, monkeypatch):
    import indexer
    from chunker import Chunk

    document = {
        "pages": [
            {"page": 1, "text": "first"},
            {"page": 2, "text": "second"},
        ]
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(document))

    import chunker
    monkeypatch.setattr(chunker, "_tokenizer", lambda: type("Tokenizer", (), {
        "__call__": lambda self, text, add_special_tokens=False, return_offsets_mapping=False: {
            "offset_mapping": [(match.start(), match.end()) for match in __import__("re").finditer(r"\S+", text)]
        },
        "encode": lambda self, text, add_special_tokens=False: list(__import__("re").finditer(r"\S+", text)),
    })())
    chunks = list(_page_chunks(path))

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page_num for chunk in chunks] == [1, 2]