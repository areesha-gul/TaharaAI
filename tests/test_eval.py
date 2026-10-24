from eval.run_eval import matches, metrics


def test_metrics_recall_at_5_and_mrr():
    questions = [
        ({"expected_doc": "A.pdf", "expected_section_id": "A.5.15"}, [
            {"doc_name": "Other", "section_id": "1.1"},
            {"doc_name": "A", "section_id": "A.5.15"},
        ]),
        ({"expected_doc": "B", "expected_section_id": "7"}, []),
    ]

    result = metrics(questions)

    assert result == {"questions": 2, "recall_at_5": 0.5, "mrr": 0.25}


def test_metrics_uses_requested_k_in_result_name():
    expected = {"expected_doc": "A", "expected_section_id": "1"}
    result = metrics([(expected, [{"doc_name": "A", "section_id": "1"}])], k=1)
    assert result["recall_at_1"] == 1.0


def test_matches_secondary_merged_section_id():
    expected = {"expected_doc": "A", "expected_section_id": "A.5.16"}
    actual = {"doc_name": "A", "section_id": "A.5.15", "section_ids": ["A.5.15", "A.5.16"]}
    assert matches(actual, expected)
