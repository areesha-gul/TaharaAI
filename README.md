# GCR / ISO RAG Pipeline

A local document pipeline for extracting compliance PDFs, indexing clause-aware chunks in PostgreSQL/pgvector, retrieving with dense or hybrid search, and generating evidence-grounded answers.

## Setup

Copy `.env.example` to `.env` and set a private PostgreSQL password. `.env` is ignored by Git. The Python client accepts `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` (with legacy `DB_NAME`, `DB_USER`, `DB_PASSWORD` fallbacks).

Start pgvector if you do not already have a PostgreSQL instance with pgvector:

```bash
docker compose up -d
```

Compose publishes PostgreSQL only on `127.0.0.1:${DB_PORT}`. If another local PostgreSQL already owns port 5432, use that database or choose another `DB_PORT`; do not point at an empty database unless you plan to re-index.

Install packages:

```bash
.venv/bin/pip install -r requirements.txt
```

For OCR, PaddleOCR also needs its supported system dependencies. The extraction code uses PyMuPDF for page classification, native text, and OCR page rendering. OCR failures are recorded per page; they do not abort later pages.

The default answer endpoint is local Ollama. Install Ollama and fetch the configured model:

```bash
ollama pull qwen2.5:7b
```

Ollama must be running for generated answers. The configured `OLLAMA_NUM_CTX=4096` and `MAX_EVIDENCE_CHARS=6000` keep evidence within the default model context. For CPU-only systems the model can be very slow; a smaller model or a hosted OpenAI-compatible endpoint may be more practical.

Initialize or migrate the database schema:

```bash
.venv/bin/python db.py
```

The migration adds page/section provenance arrays, metadata columns, full-text columns, GIN indexes, and a document-chunk uniqueness constraint. It removes duplicate rows with identical `(doc_name, page_num, chunk_index)` while retaining the newest row. Existing embeddings remain valid, but existing chunks lack the new metadata and may use page-local chunk indexes. Re-index processed documents to populate the new chunking and metadata. Indexing replaces each document's rows atomically and is safe to repeat. Before migrating a populated database, take a backup; then run `python db.py` and `python indexer.py`. The schema migration and re-index are deliberately separate operations.

## PDF processing and indexing

Put source PDFs in `documents/raw/`, then run:

```bash
.venv/bin/python triage_pipeline.py
.venv/bin/python indexer.py
```

Processed JSON newer than its source PDF is skipped. The JSON records document kind; source PDFs are not copied into duplicate type folders. Page classification uses configurable `MIN_NATIVE_CHARS` and `FULL_PAGE_IMAGE_RATIO` (defaults 50 characters and 0.70 image coverage). Native text is captured during classification; scanned pages are rendered then OCRed. Repeated header/footer lines found in page margins on more than half of pages are removed, and hyphenated line wraps are joined. Failures are recorded per page and later pages continue processing.

Indexing uses the BGE-M3 tokenizer, a target of 300 tokens, 45-token overlap, and a hard maximum of 480 tokens. Clause IDs and heading paths are retained. Per-document `metadata.json` may provide entries keyed by processed filename stem, for example:

```json
{
  "ISO_IEC-270012022-ed.3": {
    "standard": "ISO/IEC 27001",
    "edition": "2022",
    "doc_kind": "standard"
  }
}
```

A few known filenames have built-in standard/edition/document-kind mappings; top-level metadata in the processed JSON or `metadata.json` overrides those values. PDF extraction `kind` (`native`, `scanned`, `mixed`) is kept separate and is not confused with semantic `doc_kind`.

## Simple UI

A lightweight browser UI is included at `ui/index.html`. Start it with:

```bash
.venv/bin/python ui_server.py
```

Then open <http://localhost:8000> in your browser. The page lets you ask a question and shows the generated answer plus source references.

On Windows, use a Python 3.12 virtual environment and redirect package temporary files to it (the UI also stores model caches under the active virtual environment):

```powershell
py -3.12 -m venv .venv312
New-Item -ItemType Directory -Force .venv312\temp | Out-Null
$env:TEMP = "$PWD\.venv312\temp"
$env:TMP = $env:TEMP
.\.venv312\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv312\Scripts\python.exe ui_server.py
```

## Querying

Answers use dense search by default and include retrieved source records:

```bash
.venv/bin/python query.py "What requirements apply to terminated employee accounts?"
```

Inspect passages without calling the LLM:

```bash
.venv/bin/python query.py --search-only --mode hybrid --rerank "access review requirements"
```

`--rerank` and `--no-rerank` are paired flags. `--mode dense` is the default; `--mode hybrid` fuses PostgreSQL full-text/exact-clause retrieval with vector retrieval via reciprocal-rank fusion. `--expand-neighbors` adds adjacent document chunks. Results preserve cosine `score`, optional `keyword_score` and `rrf_score`, and optional `rerank_score` separately. Weak matches below `MIN_RETRIEVAL_SCORE` are dropped unless the lexical score or an exact clause match supports them; no remaining evidence returns a no-evidence answer.

Programmatic use:

```python
from query import ask, search

result = ask("What requirements apply to terminated employee accounts?", mode="hybrid")
print(result["answer"])
print(result["sources"])

hits = search("A.5.15 access rights", mode="hybrid", use_reranker=True)
```

Citation labels support `[S1]` and grouped form `[S1, S2]`. Invalid labels are replaced and returned as warnings; paragraphs without citations are reported. Model output is not a legal opinion or proof of compliance. Verify clauses against the source edition and applicable scope.

## Retrieval evaluation

Add one JSON object per line to `eval/questions.jsonl`:

```json
{"question":"What does clause A.5.15 require?","expected_doc":"ISO_IEC-270012022-ed.3","expected_section_id":"A.5.15"}
```

Run the same cases against dense, hybrid, and hybrid+rerank retrieval:

```bash
.venv/bin/python -m eval.run_eval
```

The report includes recall@5 and MRR. Leave the file empty until you have verified ground truth in the indexed source editions.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `POSTGRES_DB` | `rag_db` | PostgreSQL database |
| `POSTGRES_USER` | `rag` | PostgreSQL user |
| `POSTGRES_PASSWORD` | set in `.env` | PostgreSQL password |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Host port |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `10` | Threaded connection-pool bounds |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding and tokenizer model |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model |
| `RETRIEVE_TOP_K` / `RERANK_TOP_K` | `100` / `5` | Candidate and result counts |
| `ENABLE_RERANK` | `false` | Default reranker setting |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `300` / `45` | Token target and overlap |
| `MIN_CLAUSE_TOKENS` | `80` | Merge short sibling clauses |
| `MIN_RETRIEVAL_SCORE` | `0.35` | Initial dense cosine floor; tune with eval data |
| `MIN_KEYWORD_SCORE` | `0.01` | Hybrid lexical confidence floor |
| `MIN_KEYWORD_SCORE` | `0.01` | Hybrid lexical confidence floor |
| `RRF_K` | `60` | Reciprocal-rank fusion constant |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen2.5:7b` | Chat model |
| `LLM_TIMEOUT` | `600` seconds | Chat request timeout |
| `MAX_EVIDENCE_CHARS` | `6000` | Bound retrieved evidence prompt |
| `OLLAMA_NUM_CTX` | `4096` | Context passed to Ollama |
| `LLM_MAX_TOKENS` | `512` | Maximum generated tokens |
| `MIN_NATIVE_CHARS` | `50` | Text threshold for native pages |
| `FULL_PAGE_IMAGE_RATIO` | `0.70` | Image coverage threshold for scanned pages |
| `OCR_DPI` | `200` | Scanned page rendering resolution |
| `MIN_NATIVE_CHARS` | `50` | Text threshold for native pages |
| `FULL_PAGE_IMAGE_RATIO` | `0.70` | Image coverage threshold for scanned pages |
| `OCR_DPI` | `200` | Scanned page render resolution |
