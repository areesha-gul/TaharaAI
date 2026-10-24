"""
indexer.py — reads processed text files → chunks → embeds → stores in pgvector.

Expects the folder structure already created by your triage_pipeline:
    documents/processed/<filename>.txt   (one file per source PDF)

Usage:
    python indexer.py
    python indexer.py --processed-dir /path/to/processed
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

from chunker import chunk_pages, chunk_text
from embedder import embed_texts
from db import replace_document_chunks, init_db

load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "documents/processed"))
BATCH_SIZE    = 32   # chunks to embed + insert per batch


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_processed(path: Path) -> tuple[str, list[dict] | None]:
    """
    Supports two formats saved by extractor.py:
      - plain .txt  → read directly
      - .json with {"pages": [{"page": 1, "text": "..."}]}
    """
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        # flatten pages into one string; keep page numbers
        pages = data.get("pages", [])
        return "\n\n".join(p.get("text", "") for p in pages), pages
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, None


def _page_chunks(path: Path):
    """
    Yield Chunk objects for a single processed file.
    If the file has page-level data we chunk per page so page_num is accurate.
    Otherwise chunk the whole text at once.
    """
    raw, pages = _read_processed(path)

    if pages:
        yield from chunk_pages(pages)
    else:
        yield from chunk_text(raw)


_DOCUMENT_METADATA = {
    "ISO_IEC-270012022-ed.3": {"standard": "ISO/IEC 27001", "edition": "2022", "doc_kind": "standard"},
    "EU AI Act": {"standard": "EU AI Act", "edition": "2024", "doc_kind": "regulation"},
    "NIST.AI.100-1": {"standard": "NIST AI RMF", "edition": "1.0", "doc_kind": "framework"},
    "sama": {"standard": "SAMA Cyber Security Framework", "doc_kind": "framework"},
    "SOC-2-Compliance-Handbook_Whitepaper": {"standard": "SOC 2", "doc_kind": "guidance"},
}


def _document_metadata(path: Path, data: dict | None) -> dict:
    metadata_file = path.parent / "metadata.json"
    configured = {}
    if metadata_file.exists():
        configured = json.loads(metadata_file.read_text(encoding="utf-8")).get(path.stem, {})
    inferred = _DOCUMENT_METADATA.get(path.stem, {})
    data = data or {}
    return {
        "standard": configured.get("standard") or data.get("standard") or inferred.get("standard"),
        "edition": configured.get("edition") or data.get("edition") or inferred.get("edition"),
        "doc_kind": configured.get("doc_kind") or data.get("doc_kind") or inferred.get("doc_kind"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def index_all(processed_dir: Path):
    init_db()   # create tables if not present

    files = sorted(
        path for path in processed_dir.iterdir()
        if path.is_file() and path.name != "metadata.json" and path.suffix.lower() in {".txt", ".json"}
    ) if processed_dir.exists() else []
    if not files:
        print(f"No files found in {processed_dir}")
        return

    print(f"Indexing {len(files)} document(s) from {processed_dir}")

    for path in tqdm(files, desc="Documents"):
        doc_name = path.stem
        all_chunks = list(_page_chunks(path))

        if not all_chunks:
            print(f"  ⚠ no text extracted from {path.name}, skipping")
            replace_document_chunks(path.stem, [])
            continue

        metadata = _document_metadata(path, json.loads(path.read_text(encoding="utf-8"))
                          if path.suffix.lower() == ".json" else None)
        document_rows = []
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [c.text for c in batch]

            embeddings = embed_texts(texts)   # list of 1024-float lists

            document_rows.extend([
                {
                    "page_num":    c.page_num,
                    "page_nums":   c.page_nums,
                    "chunk_index": c.chunk_index,
                    "text":        c.text,
                    "embedding":   emb,
                    "section_id":  c.section_id,
                    "section_ids": c.section_ids,
                    "heading_path": c.heading_path,
                    **metadata,
                }
                for c, emb in zip(batch, embeddings)
            ])

        replace_document_chunks(doc_name, document_rows)

        print(f"  ✓ {path.name} — {len(all_chunks)} chunks")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        default=str(PROCESSED_DIR),
        help="Folder of extracted .txt / .json files",
    )
    args = parser.parse_args()
    index_all(Path(args.processed_dir))
