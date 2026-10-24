"""
query.py — semantic search over pgvector + cross-encoder reranking.

Two-stage:
  1. BGE-M3 dense embedding → ANN search in pgvector (top-100 candidates)
  2. BGE-Reranker-v2-M3     → rerank candidates → top-5 returned

Usage (CLI):
    python query.py "What are the requirements for access control under ISO 27001?"

Usage (as a module):
    from query import search
    results = search("your question here")
    for r in results:
        print(r["score"], r["doc_name"], r["text"][:120])
"""

import argparse
import os
from dotenv import load_dotenv

from embedder import embed_query, rerank_with_scores
from db import dense_search_chunks, expand_neighbor_chunks, keyword_search_chunks
from generator import generate_answer

load_dotenv()

RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 100))
RERANK_TOP_K   = int(os.getenv("RERANK_TOP_K",   5))
RERANK_ENABLED  = os.getenv("ENABLE_RERANK", "false").lower() in {"1", "true", "yes", "on"}
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
MIN_KEYWORD_SCORE = float(os.getenv("MIN_KEYWORD_SCORE", "0.01"))
RRF_K = int(os.getenv("RRF_K", "60"))


def _rrf(dense: list[dict], keyword: list[dict], top_k: int) -> list[dict]:
    """Fuse dense and keyword ranks while preserving each retriever's scores."""
    merged = {}
    for result_list, rank_field in ((dense, "dense_rank"), (keyword, "keyword_rank")):
        for rank, result in enumerate(result_list, 1):
            item = merged.setdefault(result["id"], dict(result))
            item[rank_field] = rank
            if rank_field == "dense_rank":
                item["score"] = result.get("score")
            else:
                item["keyword_score"] = result.get("score", 0.0)
                item["exact_clause_match"] = bool(
                    item.get("exact_clause_match") or result.get("exact_clause_match")
                )
    for item in merged.values():
        item["rrf_score"] = sum(
            1.0 / (RRF_K + item[field]) for field in ("dense_rank", "keyword_rank") if field in item
        )
        item["score"] = item.get("score", 0.0)
        item["exact_clause_match"] = bool(item.get("exact_clause_match", False))
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]


def search(
    query: str,
    retrieve_k: int = RETRIEVE_TOP_K,
    rerank_k: int = RERANK_TOP_K,
    use_reranker: bool = RERANK_ENABLED,
    mode: str = "dense",
    expand_neighbors: bool = False,
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> list[dict]:
    """
    Full semantic search pipeline:

        query text
          → BGE-M3 dense embedding
          → pgvector ANN (cosine) → top-{retrieve_k} candidates
          → optionally rerank candidates → top-{rerank_k} results

    Returns a list of dicts:
        {id, doc_name, page_num, chunk_index, text, score}
    """

    # 1. embed the query
    q_vec = embed_query(query)

    # 2. Retrieve dense results, optionally fused with lexical search.
    dense = dense_search_chunks(q_vec, top_k=retrieve_k)
    if mode == "dense":
        candidates = dense
    elif mode == "hybrid":
        keyword = keyword_search_chunks(query, top_k=retrieve_k)
        candidates = _rrf(dense, keyword, retrieve_k)
    else:
        raise ValueError("mode must be 'dense' or 'hybrid'")
    if not candidates:
        return []

    # Keep confident semantic matches, exact clause hits, and strong lexical hits.
    candidates = [
        candidate for candidate in candidates
        if candidate.get("score", 0.0) >= min_score
        or candidate.get("exact_clause_match")
        or candidate.get("keyword_score", 0.0) >= MIN_KEYWORD_SCORE
    ]
    if not candidates:
        return []

    if use_reranker:
        passages = [candidate["text"] for candidate in candidates]
        scored = rerank_with_scores(query, passages, top_k=rerank_k)
        results = []
        for index, rerank_score in scored:
            candidate = dict(candidates[index])
            candidate["rerank_score"] = rerank_score
            results.append(candidate)
    else:
        results = candidates[:rerank_k]

    if expand_neighbors and results:
        results = expand_neighbor_chunks(results)
    return results


def ask(
    question: str,
    retrieve_k: int = RETRIEVE_TOP_K,
    rerank_k: int = RERANK_TOP_K,
    use_reranker: bool = RERANK_ENABLED,
    mode: str = "dense",
    expand_neighbors: bool = False,
) -> dict:
    """Retrieve evidence and generate a grounded answer with source records."""
    results = search(
        question,
        retrieve_k=retrieve_k,
        rerank_k=rerank_k,
        use_reranker=use_reranker,
        mode=mode,
        expand_neighbors=expand_neighbors,
    )
    generated = generate_answer(question, results)
    return {**generated, "retrieved": results}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(query: str, results: list[dict]):
    print(f"\nQuery: {query}\n{'─'*60}")
    if not results:
        print("No results found.")
        return
    for rank, r in enumerate(results, 1):
        source = r["doc_name"]
        if r.get("page_num"):
            source += f"  p.{r['page_num']}"
        score_text = f"cosine: {r.get('score', 0.0):.4f}"
        if r.get("keyword_score") is not None:
            score_text += f", keyword: {r['keyword_score']:.4f}"
        if r.get("rrf_score") is not None:
            score_text += f", RRF: {r['rrf_score']:.4f}"
        if r.get("rerank_score") is not None:
            score_text += f", rerank: {r['rerank_score']:.4f}"
        print(f"\n[{rank}]  {source}  ({score_text})")
        print(r["text"][:400].strip())
        if len(r["text"]) > 400:
            print("  …")
    print()


def _print_answer(result: dict):
    print(f"\nAnswer\n{'─'*60}\n{result['answer']}\n")
    for warning in result.get("warnings", []):
        print(f"Citation warning: {warning}")
    if result["sources"]:
        print("Sources")
        for source in result["sources"]:
            location = source.get("doc_name") or "Unknown document"
            page_nums = source.get("page_nums") or ([source["page_num"]] if source.get("page_num") else [])
            if page_nums:
                location += f", p. {', '.join(str(page) for page in page_nums)}"
            if source.get("section_ids"):
                location += f", section {', '.join(source['section_ids'])}"
            print(f"[{source['id']}] {location}")
            if source.get("chunk_index") is not None:
                print(f"    chunk {source['chunk_index']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounded RAG over indexed compliance documents")
    parser.add_argument("query", nargs="+", help="Your question")
    parser.add_argument("--top",  type=int, default=RERANK_TOP_K,   help="Results to return")
    parser.add_argument("--pool", type=int, default=RETRIEVE_TOP_K, help="ANN candidates before reranking")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=RERANK_ENABLED,
                        help="Use the cross-encoder reranker (or --no-rerank)")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense",
                        help="Retrieval mode")
    parser.add_argument("--expand-neighbors", action="store_true",
                        help="Include adjacent chunks for each top result")
    parser.add_argument("--search-only", action="store_true",
                        help="Show retrieved passages without generating an answer")
    args = parser.parse_args()

    q       = " ".join(args.query)
    results = search(
        q,
        retrieve_k=args.pool,
        rerank_k=args.top,
        use_reranker=args.rerank,
        mode=args.mode,
        expand_neighbors=args.expand_neighbors,
    )
    if args.search_only:
        _print_results(q, results)
    else:
        _print_answer(generate_answer(q, results))
