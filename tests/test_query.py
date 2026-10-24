import unittest
import argparse
from unittest.mock import patch

import query


class QueryTests(unittest.TestCase):
    def test_cli_rerank_has_positive_and_negative_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
        self.assertTrue(parser.parse_args(["--rerank"]).rerank)
        self.assertFalse(parser.parse_args(["--no-rerank"]).rerank)

    def test_rrf_merges_dense_and_keyword_ranks_without_overwriting_cosine_score(self):
        dense = [{"id": 1, "doc_name": "A", "score": 0.72, "text": "one"}]
        keyword = [{"id": 1, "doc_name": "A", "score": 0.4, "text": "one"}]

        result = query._rrf(dense, keyword, top_k=5)[0]

        self.assertEqual(result["score"], 0.72)
        self.assertEqual(result["keyword_score"], 0.4)
        self.assertGreater(result["rrf_score"], 0)

    def test_rrf_preserves_keyword_exact_clause_match(self):
        result = query._rrf(
            [{"id": 1, "score": 0.2, "text": "A.5.15"}],
            [{"id": 1, "score": 0.0, "exact_clause_match": True, "text": "A.5.15"}],
            top_k=5,
        )[0]
        self.assertTrue(result["exact_clause_match"])

    def test_dense_search_filters_below_minimum_score(self):
        with patch.object(query, "embed_query", return_value=[0.0]):
            with patch.object(query, "dense_search_chunks", return_value=[
                {"id": 1, "score": 0.2, "text": "weak"},
                {"id": 2, "score": 0.8, "text": "strong"},
            ]):
                results = query.search("question", min_score=0.35)

        self.assertEqual([result["id"] for result in results], [2])

    def test_ask_returns_generated_answer_and_retrieved_chunks(self):
        chunks = [{"doc_name": "Policy", "text": "Remove access after termination."}]
        generated = {"answer": "Remove access [S1].", "sources": [{"id": "S1"}]}

        with patch.object(query, "search", return_value=chunks) as search:
            with patch.object(query, "generate_answer", return_value=generated) as generate:
                result = query.ask("What about leavers?", retrieve_k=20, rerank_k=3)

        search.assert_called_once_with(
            "What about leavers?", retrieve_k=20, rerank_k=3,
            use_reranker=query.RERANK_ENABLED, mode="dense", expand_neighbors=False
        )
        generate.assert_called_once_with("What about leavers?", chunks)
        self.assertEqual(result["answer"], generated["answer"])
        self.assertEqual(result["retrieved"], chunks)


if __name__ == "__main__":
    unittest.main()