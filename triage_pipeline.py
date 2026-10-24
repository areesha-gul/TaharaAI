"""
triage_pipeline.py — runs the full document preparation pipeline.

For every PDF in documents/raw/:
    1. Classify   → native / scanned / mixed
    2. Extract    → write JSON to documents/processed/
    3. Log        → write per-file result to documents/logs/

Usage:
    python triage_pipeline.py
    python triage_pipeline.py --raw-dir /path/to/pdfs
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from classifier import classify_pdf
from extractor import extract_and_save

# ── directory layout ──────────────────────────────────────────────────────────

DOCS_ROOT = Path("documents")

RAW_DIR       = DOCS_ROOT / "raw"
PROCESSED_DIR = DOCS_ROOT / "processed"
LOGS_DIR      = DOCS_ROOT / "logs"

# created automatically
for _d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── per-file processing ───────────────────────────────────────────────────────

def process_one(pdf_path: Path) -> dict:
    """
    Run classify → route → extract for a single PDF.
    Returns a log record dict.
    """
    t0 = time.perf_counter()
    record: dict = {"file": pdf_path.name, "status": "ok"}
    out_path = PROCESSED_DIR / f"{pdf_path.stem}.json"
    if out_path.exists() and out_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        record.update(status="skipped_up_to_date", output=str(out_path))
        return record

    # 1. classify
    try:
        cr = classify_pdf(pdf_path)
        record["kind"] = cr.kind
        record["pages"] = len(cr.pages)
        if cr.error:
            record["status"] = "classify_error"
            record["error"]  = cr.error
            return record
    except Exception as e:
        record.update(status="classify_error", error=str(e))
        return record

    # 2. Extract; the detected kind is retained in the JSON rather than copying the PDF.
    try:
        out_path = extract_and_save(pdf_path, cr, PROCESSED_DIR)
        record["output"] = str(out_path)
    except Exception as e:
        record.update(status="extract_error", error=str(e))
        return record

    record["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return record


# ── main ──────────────────────────────────────────────────────────────────────

def run(raw_dir: Path):
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found in %s", raw_dir)
        return

    log.info("Found %d PDF(s) in %s", len(pdfs), raw_dir)

    summary = {"total": len(pdfs), "ok": 0, "skipped": 0, "errors": 0, "files": []}

    for pdf in tqdm(pdfs, desc="Processing"):
        record = process_one(pdf)
        summary["files"].append(record)

        if record["status"] == "ok":
            summary["ok"] += 1
            log.info("  ✓  %-40s  %-8s  %d pages  %.1fs",
                     pdf.name, record["kind"], record.get("pages", 0), record.get("elapsed_s", 0))
        elif record["status"] == "skipped_up_to_date":
            summary["skipped"] += 1
            log.info("  -  %-40s  processed output is current", pdf.name)
        else:
            summary["errors"] += 1
            log.error("  ✗  %-40s  %s  %s",
                      pdf.name, record["status"], record.get("error", ""))

    # write summary log
    log_path = LOGS_DIR / "triage_run.json"
    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("")
    log.info("Done: %d ok, %d errors  →  log: %s", summary["ok"], summary["errors"], log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify, route, and extract all PDFs")
    parser.add_argument("--raw-dir", default=str(RAW_DIR),
                        help="Folder of raw PDFs (default: documents/raw)")
    args = parser.parse_args()
    run(Path(args.raw_dir))
