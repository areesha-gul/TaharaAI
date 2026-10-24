from unittest.mock import patch

import embedder


def test_rerank_empty_candidates_does_not_load_model():
    with patch.object(embedder, "_rerank_model") as model:
        assert embedder.rerank("question", []) == []
        model.assert_not_called()


def test_rerank_single_candidate_handles_scalar_score():
    model = type("Model", (), {"compute_score": lambda self, pairs, normalize: 0.7})()
    with patch.object(embedder, "_rerank_model", return_value=model):
        assert embedder.rerank("question", ["passage"]) == [0]
        assert embedder.rerank_with_scores("question", ["passage"]) == [(0, 0.7)]


def test_rerank_handles_scalar_model_score():
    model = type("Model", (), {"compute_score": lambda self, pairs, normalize: 0.7})()
    with patch.object(embedder, "_rerank_model", return_value=model):
        assert embedder.rerank("question", ["first", "second"], top_k=1) == [0]


def test_rerank_with_scores_preserves_scalar_score_for_multiple_candidates():
    model = type("Model", (), {"compute_score": lambda self, pairs, normalize: 0.7})()
    with patch.object(embedder, "_rerank_model", return_value=model):
        assert embedder.rerank_with_scores("question", ["first", "second"], top_k=2) == [(0, 0.7)]