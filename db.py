"""Pooled PostgreSQL/pgvector access, schema migration, and retrieval queries."""

import os
import re

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()

_POOL = None


def _required_setting(primary: str, legacy: str) -> str:
    value = os.getenv(primary) or os.getenv(legacy)
    if not value:
        raise RuntimeError(f"Set {primary} in .env")
    return value


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadedConnectionPool(
            minconn=int(os.getenv("DB_POOL_MIN", "1")),
            maxconn=int(os.getenv("DB_POOL_MAX", "10")),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=_required_setting("POSTGRES_DB", "DB_NAME"),
            user=_required_setting("POSTGRES_USER", "DB_USER"),
            password=_required_setting("POSTGRES_PASSWORD", "DB_PASSWORD"),
        )
    return _POOL


def get_conn():
    """Borrow a connection from the process-wide threaded pool."""
    return _get_pool().getconn()


def put_conn(conn):
    """Return a borrowed connection to the pool in a clean transaction state."""
    if conn is None:
        return
    if conn.closed:
        _get_pool().putconn(conn, close=True)
        return
    if conn.status != psycopg2.extensions.STATUS_READY:
        conn.rollback()
    _get_pool().putconn(conn)


def close_pool():
    """Close all pooled connections, primarily for process shutdown/tests."""
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None


def _halfvec_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL,
    page_num INT,
    page_nums INT[] NOT NULL DEFAULT '{}',
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding halfvec(1024) NOT NULL,
    section_id TEXT,
    section_ids TEXT[] NOT NULL DEFAULT '{}',
    heading_path TEXT,
    standard TEXT,
    edition TEXT,
    doc_kind TEXT,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_nums INT[] NOT NULL DEFAULT '{}';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_ids TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS heading_path TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS standard TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS edition TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_kind TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED;

DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'chunks'
          AND column_name = 'search_vector'
          AND COALESCE(generation_expression, '') NOT LIKE '%regconfig%'
    ) THEN
        DROP INDEX IF EXISTS chunks_search_vector_idx;
        ALTER TABLE chunks DROP COLUMN search_vector;
        ALTER TABLE chunks ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED;
    END IF;
END $$;

UPDATE chunks
SET section_ids = ARRAY[section_id]
WHERE section_id IS NOT NULL AND cardinality(section_ids) = 0;

UPDATE chunks
SET page_nums = ARRAY[page_num]
WHERE page_num IS NOT NULL AND cardinality(page_nums) = 0;

DELETE FROM chunks older
USING chunks newer
WHERE older.doc_name = newer.doc_name
  AND older.page_num IS NOT DISTINCT FROM newer.page_num
  AND older.chunk_index = newer.chunk_index
  AND older.id < newer.id;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chunks_document_chunk_unique'
          AND conrelid = 'chunks'::regclass
    ) THEN
        ALTER TABLE chunks ADD CONSTRAINT chunks_document_chunk_unique
            UNIQUE NULLS NOT DISTINCT (doc_name, page_num, chunk_index);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_section_id_idx ON chunks (section_id);
CREATE INDEX IF NOT EXISTS chunks_section_ids_idx ON chunks USING gin (section_ids);
"""


def init_db():
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
    finally:
        put_conn(conn)
    print("schema ready")


def replace_document_chunks(doc_name: str, rows: list[dict]):
    """Atomically replace all indexed chunks for one document."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_name = %s", (doc_name,))
                if rows:
                    execute_values(
                        cur,
                        """
                        INSERT INTO chunks (
                            doc_name, page_num, page_nums, chunk_index, text, embedding,
                            section_id, section_ids, heading_path, standard, edition, doc_kind
                        ) VALUES %s
                        """,
                        [
                            (
                                doc_name,
                                row.get("page_num"),
                                list(row.get("page_nums") or ([row["page_num"]] if row.get("page_num") else [])),
                                row["chunk_index"],
                                row["text"],
                                _halfvec_literal(row["embedding"]),
                                row.get("section_id"),
                                list(row.get("section_ids") or ([row["section_id"]] if row.get("section_id") else [])),
                                row.get("heading_path"),
                                row.get("standard"),
                                row.get("edition"),
                                row.get("doc_kind"),
                            )
                            for row in rows
                        ],
                        template="(%s, %s, %s, %s, %s, %s::halfvec, %s, %s, %s, %s, %s, %s)",
                    )
    finally:
        put_conn(conn)


def insert_chunks(rows: list[dict]):
    """Compatibility wrapper; replace each document atomically."""
    documents = {}
    for row in rows:
        documents.setdefault(row["doc_name"], []).append(row)
    for doc_name, document_rows in documents.items():
        replace_document_chunks(doc_name, document_rows)


def _fetch_all(cur) -> list[dict]:
    columns = [description[0] for description in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def dense_search_chunks(query_embedding: list[float], top_k: int = 100) -> list[dict]:
    """Search nearest chunks by cosine similarity; raise HNSW ef_search per query."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                ef_search = max(int(top_k), 100)
                cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                vector = _halfvec_literal(query_embedding)
                cur.execute(
                    """
                    SELECT id, doc_name, page_num, page_nums, chunk_index, text,
                           section_id, section_ids, heading_path, standard, edition, doc_kind,
                           1 - (embedding <=> %s::halfvec) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::halfvec
                    LIMIT %s
                    """,
                    (vector, vector, top_k),
                )
                return _fetch_all(cur)
    finally:
        put_conn(conn)


def search_chunks(query_embedding: list[float], top_k: int = 100) -> list[dict]:
    """Backward-compatible alias for dense vector search."""
    return dense_search_chunks(query_embedding, top_k)


def _extract_clause_ids(query: str) -> list[str]:
    return list(dict.fromkeys(value.upper() for value in re.findall(
        r"(?<![A-Za-z0-9])(?:[A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)+)(?![A-Za-z0-9])",
        query,
        flags=re.IGNORECASE,
    )))


def _clause_patterns(clause_ids: list[str]) -> list[str]:
    return [
        rf"(^|[^A-Za-z0-9.]){re.escape(clause_id)}($|[^A-Za-z0-9.]|\.(?!\d))"
        for clause_id in clause_ids
    ]


def _keyword_search_cursor(cur, query, top_k, clause_ids, clause_patterns):
    cur.execute(
        """
        SELECT id, doc_name, page_num, page_nums, chunk_index, text,
               section_id, section_ids, heading_path, standard, edition, doc_kind,
               ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS score,
               (section_ids && %s OR section_id = ANY(%s) OR text ~* ANY(%s)) AS exact_clause_match
        FROM chunks
        WHERE search_vector @@ plainto_tsquery('english', %s)
           OR section_ids && %s
           OR section_id = ANY(%s)
           OR text ~* ANY(%s)
        ORDER BY (section_ids && %s OR section_id = ANY(%s) OR text ~* ANY(%s)) DESC,
                 score DESC
        LIMIT %s
        """,
        (
            query,
            clause_ids,
            clause_ids,
            clause_patterns,
            query,
            clause_ids,
            clause_ids,
            clause_patterns,
            clause_ids,
            clause_ids,
            clause_patterns,
            top_k,
        ),
    )
    return _fetch_all(cur)


def keyword_search_chunks(query: str, top_k: int = 100) -> list[dict]:
    """Search full text plus exact clause IDs, with keyword rank as score."""
    clause_ids = _extract_clause_ids(query)
    clause_patterns = _clause_patterns(clause_ids)
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                return _keyword_search_cursor(cur, query, top_k, clause_ids, clause_patterns)
    finally:
        put_conn(conn)


def expand_neighbor_chunks(hits: list[dict]) -> list[dict]:
    """Include adjacent document-global chunks for additional context."""
    if not hits:
        return []
    groups = {}
    for hit in hits:
        groups.setdefault(hit["doc_name"], set()).update(
            hit["chunk_index"] + offset for offset in (-1, 0, 1)
        )
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                neighbors = []
                for doc_name, indexes in groups.items():
                    cur.execute(
                        """
                        SELECT id, doc_name, page_num, page_nums, chunk_index, text,
                               section_id, section_ids, heading_path, standard, edition, doc_kind
                        FROM chunks
                        WHERE doc_name = %s AND chunk_index = ANY(%s)
                        ORDER BY chunk_index
                        """,
                        (doc_name, list(indexes)),
                    )
                    neighbors.extend(_fetch_all(cur))
        by_key = {(row["doc_name"], row["chunk_index"]): row for row in neighbors}
        hit_details = {(row["doc_name"], row["chunk_index"]): row for row in hits}
        output, seen = [], set()
        for hit in hits:
            for offset in (-1, 0, 1):
                key = (hit["doc_name"], hit["chunk_index"] + offset)
                if key in by_key and key not in seen:
                    row = dict(by_key[key])
                    row.update({
                        field: hit_details[key][field]
                        for field in ("score", "keyword_score", "rrf_score", "rerank_score")
                        if key in hit_details and field in hit_details[key]
                    })
                    row["is_neighbor"] = key not in hit_details
                    output.append(row)
                    seen.add(key)
        return output
    finally:
        put_conn(conn)


if __name__ == "__main__":
    init_db()
