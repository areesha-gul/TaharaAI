"""
embedder.py — BGE-M3 dense embeddings + BGE-Reranker-v2-M3.

Loaded once as module-level singletons so the model isn't
re-initialised on every call.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL  = os.getenv("EMBED_MODEL",  "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


# ── lazy singletons ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _embed_model():
    from FlagEmbedding import BGEM3FlagModel
    import torch
    print(f"  loading embedder: {EMBED_MODEL}")
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=torch.cuda.is_available())


@lru_cache(maxsize=1)
def _rerank_model():
    from FlagEmbedding import FlagReranker
    import torch
    print(f"  loading reranker: {RERANK_MODEL}")
    return FlagReranker(RERANK_MODEL, use_fp16=torch.cuda.is_available())


# ── public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with BGE-M3.
    Returns a list of 1024-float dense vectors.
    """
    model = _embed_model()
    out = model.encode(
        texts,
        batch_size=12,
        max_length=512,
        return_dense=True,
        return_sparse=False,   # keep it fast; enable for hybrid search later
        return_colbert_vecs=False,
    )
    # out["dense_vecs"] is a numpy array — convert to plain lists for psycopg2
    return out["dense_vecs"].tolist()


def embed_query(query: str) -> list[float]:
    """Single query — wraps embed_texts for convenience."""
    return embed_texts([query])[0]


def rerank_with_scores(query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
    """
    Score each (query, passage) pair and return the indices of the
    top_k passages, ordered best-first.
    """
    if not passages or top_k <= 0:
        return []

    model = _rerank_model()
    pairs  = [[query, p] for p in passages]
    scores = model.compute_score(pairs, normalize=True)  # list of floats
    if not hasattr(scores, "__iter__"):
        scores = [float(scores)]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(index, float(scores[index])) for index in ranked[:top_k]]


def rerank(query: str, passages: list[str], top_k: int = 5) -> list[int]:
    """Return best-first passage indexes for backwards compatibility."""
    return [index for index, _score in rerank_with_scores(query, passages, top_k)]
