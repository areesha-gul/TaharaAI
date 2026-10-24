import json
import unittest
from unittest.mock import patch

import generator


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class GeneratorTests(unittest.TestCase):
    def test_no_evidence_returns_without_calling_model(self):
        with patch.object(generator, "urlopen") as open_url:
            result = generator.generate_answer("Which clause applies?", [])

        self.assertEqual(result["sources"], [])
        self.assertIn("could not find relevant evidence", result["answer"])
        open_url.assert_not_called()

    def test_answer_uses_source_label_and_returns_source_record(self):
        retrieved = [{
            "doc_name": "Access Standard",
            "page_num": 7,
            "chunk_index": 2,
            "section_id": "A.5.15",
            "section_ids": ["A.5.15", "A.5.16"],
            "score": 0.91,
            "text": "Access rights should be removed when employment ends.",
        }]
        payload = {"choices": [{"message": {"content": "Remove the account [S1]."}}]}
        with patch.object(generator, "urlopen", return_value=FakeResponse(payload)) as open_url:
            result = generator.generate_answer("What about leavers?", retrieved)

        request = open_url.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertIn("[S1] Access Standard, p. 7, section A.5.15, A.5.16", request_body["messages"][1]["content"])
        self.assertEqual(result["answer"], "Remove the account [S1].")
        self.assertEqual(result["sources"][0]["doc_name"], "Access Standard")
        self.assertEqual(result["sources"][0]["section_id"], "A.5.15")

    def test_unknown_citation_label_is_flagged(self):
        retrieved = [{"doc_name": "Policy", "text": "Use unique accounts."}]
        payload = {"choices": [{"message": {"content": "Use unique accounts [S2]."}}]}
        with patch.object(generator, "urlopen", return_value=FakeResponse(payload)):
            result = generator.generate_answer("What is required?", retrieved)

        self.assertIn("lacked valid source citations", result["answer"])
        self.assertTrue(any("Unknown citation" in warning for warning in result["warnings"]))

    def test_generation_timeout_has_clear_diagnostic(self):
        retrieved = [{"doc_name": "Policy", "text": "Use unique accounts."}]
        with patch.object(generator, "urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, "did not finish within 600 seconds"):
                generator.generate_answer("What is required?", retrieved)

    def test_ollama_request_sets_context_and_generation_limit(self):
        payload = {"choices": [{"message": {"content": "Answer [S1]."}}]}
        retrieved = [{"doc_name": "Policy", "text": "A policy excerpt."}]
        with patch.object(generator, "urlopen", return_value=FakeResponse(payload)) as open_url:
            generator.generate_answer("Question?", retrieved)

        sent = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        if "localhost:11434" in generator.LLM_BASE_URL or "127.0.0.1:11434" in generator.LLM_BASE_URL:
            self.assertEqual(sent["options"]["num_ctx"], generator.OLLAMA_NUM_CTX)
        self.assertEqual(sent["max_tokens"], generator.LLM_MAX_TOKENS)

    def test_grouped_citations_validate_all_labels(self):
        answer, warnings = generator.validate_citations("Claim [S1, S2].", 2)
        self.assertEqual(answer, "Claim [S1, S2].")
        self.assertEqual(warnings, [])

    def test_unknown_grouped_citation_and_uncited_line_are_omitted(self):
        answer, warnings = generator.validate_citations("Fact [S1, S3].\n\nUncited note.", 1)
        self.assertEqual(answer, "")
        self.assertTrue(any("Unknown citation" in warning for warning in warnings))
        self.assertTrue(any("no valid source citation" in warning for warning in warnings))

    def test_exact_clause_question_disclaims_missing_clause_metadata(self):
        retrieved = [{"doc_name": "Policy", "text": "Remove access when employment ends."}]
        payload = {"choices": [{"message": {"content": "Review access promptly [S1]."}}]}
        with patch.object(generator, "urlopen", return_value=FakeResponse(payload)):
            result = generator.generate_answer("Which exact clause applies?", retrieved)

        self.assertIn("I cannot verify an exact clause from the retrieved evidence.", result["answer"])

    def test_uncited_lines_are_removed(self):
        answer, warnings = generator.validate_citations("Supported [S1].\nUnsupported claim.", 1)
        self.assertEqual(answer, "Supported [S1].")
        self.assertTrue(any("no valid source citation" in warning for warning in warnings))

    def test_ollama_request_sets_context_and_generation_limit(self):
        payload = {"choices": [{"message": {"content": "Answer [S1]."}}]}
        retrieved = [{"doc_name": "Policy", "text": "A policy excerpt."}]
        with patch.object(generator, "urlopen", return_value=FakeResponse(payload)) as open_url:
            generator.generate_answer("Question?", retrieved)

        sent = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        if "localhost:11434" in generator.LLM_BASE_URL or "127.0.0.1:11434" in generator.LLM_BASE_URL:
            self.assertEqual(sent["options"]["num_ctx"], generator.OLLAMA_NUM_CTX)
        self.assertEqual(sent["max_tokens"], generator.LLM_MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()