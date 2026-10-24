"""Evaluate retrieval recall@5 and MRR against user-supplied JSONL ground truth."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from query import RETRIEVE_TOP_K, search

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        required = {"question", "expected_doc", "expected_section_id"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Line {line_number} missing fields: {', '.join(sorted(missing))}")
        questions.append(item)
    return questions


def matches(result: dict, expected: dict) -> bool:
    expected_doc = Path(expected["expected_doc"]).stem.casefold()
    actual_doc = Path(str(result.get("doc_name", ""))).stem.casefold()
    actual_sections = {
        section.strip().casefold()
        for section in (result.get("section_ids") or [])
    }
    actual_sections.add(str(result.get("section_id", "")).casefold())
    return actual_doc == expected_doc and str(expected["expected_section_id"]).casefold() in actual_sections


def metrics(results_by_question: list[tuple[dict, list[dict]]], k: int = 5) -> dict:
    if not results_by_question:
        return {"questions": 0, f"recall_at_{k}": 0.0, "mrr": 0.0}
    found = 0
    reciprocal_ranks = []
    for expected, results in results_by_question:
        rank = next((i for i, result in enumerate(results[:k], 1) if matches(result, expected)), None)
        if rank is not None:
            found += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    count = len(results_by_question)
    return {"questions": count, f"recall_at_{k}": found / count, "mrr": sum(reciprocal_ranks) / count}


def run(path: Path = QUESTIONS_PATH, top_k: int = 5) -> dict:
    questions = load_questions(path)
    if not questions:
        print(f"No evaluation cases yet. Add JSONL rows to {path}.")
        return {}
    modes = {
        "dense": ("dense", False),
        "hybrid": ("hybrid", False),
        "hybrid+rerank": ("hybrid", True),
    }
    reports = {}
    for name, (mode, use_reranker) in modes.items():
        results = []
        for item in questions:
            hits = search(
                item["question"],
                retrieve_k=RETRIEVE_TOP_K,
                rerank_k=top_k,
                use_reranker=use_reranker,
                mode=mode,
            )
            results.append((item, hits))
        reports[name] = metrics(results, k=top_k)
        print(
            f"{name:14} recall@{top_k}={reports[name][f'recall_at_{top_k}']:.3f} "
            f"MRR={reports[name]['mrr']:.3f}"
        )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    run(args.questions, args.top)
