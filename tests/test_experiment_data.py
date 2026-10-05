from __future__ import annotations

import json
import math
import sys
import types

import pytest

from megamem.experiments.data import load_questions, load_records, select_questions
from megamem.experiments.evaluation import Evaluator, aggregate, resolve_adapter


def _enterprise_questions():
    return [
        {"question_id": f"q{index:03d}", "question": f"Question {index}", "gold_answer": None}
        for index in range(500)
    ]


def _split_file(tmp_path, **updates):
    questions = _enterprise_questions()
    manifest = {
        "seed": 42,
        "dev_question_ids": [row["question_id"] for row in questions[:100]],
        "test_question_ids": [row["question_id"] for row in questions[100:]],
    }
    manifest.update(updates)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _question():
    return {
        "question_id": "q1",
        "question": "What is required?",
        "gold_answer": "two signatures",
        "expected_doc_ids": ["a", "b"],
        "question_type": "fact",
    }


def _prediction(**updates):
    row = {
        "question_id": "q1",
        "prediction": "two signatures",
        "loaded_doc_ids": ["a", "b", "c"],
        "reported_doc_ids": ["a"],
        "status": "ok",
        "errors": [],
    }
    row.update(updates)
    return row


def _adapter(monkeypatch, function):
    module = types.ModuleType("megamem_test_adapter")
    module.score = function
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:score"


def _evaluator(adapter=None):
    return Evaluator({"evaluation": {"adapter": adapter, "score_scale": 100, "allow_unscored": True}})


@pytest.mark.parametrize("extension,content", [
    (".jsonl", '{"id":"one","query":"Question?"}\n\n'),
    (".json", '[{"id":"one","query":"Question?"}]'),
    (".json", '{"questions":[{"id":"one","query":"Question?"}]}'),
])
def test_load_questions_preserves_missing_gold(tmp_path, extension, content):
    path = tmp_path / f"questions{extension}"
    path.write_text(content, encoding="utf-8")
    rows = load_questions(path)
    assert rows[0]["question_id"] == "one"
    assert rows[0]["question"] == "Question?"
    assert rows[0]["gold_answer"] is None
    assert rows[0]["expected_doc_ids"] is None
    assert rows[0]["answer_facts"] is None
    assert rows[0]["gold_chunks"] is None


def test_jsonl_records_stream_before_invalid_later_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id":"first"}\ninvalid\n', encoding="utf-8")
    iterator = load_records(path)
    assert next(iterator) == {"id": "first"}
    with pytest.raises(ValueError, match="invalid JSON"):
        next(iterator)


def test_load_questions_rejects_duplicate_normalized_ids(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"question_id": 1, "question": "First?"},
        {"question_id": "1", "question": "Second?"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question_id"):
        load_questions(path)


def test_load_records_parquet_is_lazy_and_uses_batches(tmp_path, monkeypatch):
    opened = []
    batches = []
    arrow = types.ModuleType("pyarrow")
    arrow.__path__ = []
    parquet = types.ModuleType("pyarrow.parquet")

    class FakeBatch:
        def to_pylist(self):
            return [{"id": "row"}]

    class FakeParquetFile:
        def __init__(self, path):
            opened.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_batches(self, batch_size):
            batches.append(batch_size)
            yield FakeBatch()

    parquet.ParquetFile = FakeParquetFile
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    iterator = load_records(tmp_path / "data.parquet")
    assert opened == []
    assert list(iterator) == [{"id": "row"}]
    assert batches == [1024]


def test_enterprise_split_keeps_development_and_validation_disjoint(tmp_path):
    questions = _enterprise_questions()
    path = _split_file(tmp_path)
    dev = select_questions(questions, path, "dev", 100)
    validation = select_questions(questions, path, "validation", 400)
    assert {row["question_id"] for row in dev}.isdisjoint(row["question_id"] for row in validation)
    assert len(select_questions(questions, path, "all", 500)) == 500
    with pytest.raises(ValueError, match="expected exactly 500"):
        select_questions(questions, path, "validation", 500)


def test_enterprise_split_rejects_overlap(tmp_path):
    ids = [row["question_id"] for row in _enterprise_questions()[100:]]
    ids[-1] = "q000"
    path = _split_file(tmp_path, test_question_ids=ids)
    with pytest.raises(ValueError, match="overlap"):
        select_questions(_enterprise_questions(), path, "validation", 400)


def test_enterprise_split_rejects_duplicate_manifest_ids(tmp_path):
    ids = [row["question_id"] for row in _enterprise_questions()[:100]]
    ids[-1] = ids[0]
    path = _split_file(tmp_path, dev_question_ids=ids)
    with pytest.raises(ValueError, match="duplicate identifiers"):
        select_questions(_enterprise_questions(), path, "dev", 100)


def test_enterprise_split_requires_complete_matching_question_ids(tmp_path):
    path = _split_file(tmp_path)
    with pytest.raises(ValueError, match="do not match manifest"):
        select_questions(_enterprise_questions()[:-1], path, "dev", 100)


def test_enterprise_split_rejects_wrong_size_and_seed(tmp_path):
    path = _split_file(tmp_path, dev_question_ids=["q000"])
    with pytest.raises(ValueError, match="100 development and 400 validation"):
        select_questions(_enterprise_questions(), path, "dev", 1)
    path = _split_file(tmp_path, seed=7)
    with pytest.raises(ValueError, match="seed 42"):
        select_questions(_enterprise_questions(), path, "dev", 100)


def test_explicit_sampling_is_deterministic_and_never_silently_truncates(tmp_path):
    questions = _enterprise_questions()
    path = _split_file(tmp_path)
    first = select_questions(questions, path, "all", 100, sample_size=100, seed=42)
    second = select_questions(questions, path, "all", 100, sample_size=100, seed=42)
    assert first == second
    assert len({row["question_id"] for row in first}) == 100
    with pytest.raises(ValueError, match="expected exactly 100"):
        select_questions(questions, path, "all", 100)
    with pytest.raises(ValueError, match="Cannot sample"):
        select_questions(questions, path, "dev", 101, sample_size=101)


def test_external_task_selection_requires_no_enterprise_split():
    questions = _enterprise_questions()[:150]
    assert len(select_questions(questions, None, "task", 150)) == 150
    with pytest.raises(ValueError, match="manifest is required"):
        select_questions(questions, None, "validation", 150)
    with pytest.raises(ValueError, match="expected exactly 200"):
        select_questions(questions, None, "all", 200)


def test_unscored_evaluation_has_no_invented_official_scores():
    result = _evaluator().evaluate(_question(), _prediction())
    assert result["evaluation_status"] == "unscored"
    assert all(result[key] is None for key in ("correctness", "completeness", "overall", "invalid_documents"))
    assert result["lexical_f1"] == 1.0
    assert result["loaded_document_recall"] == 1.0
    assert result["reported_document_recall"] == 0.5


def test_missing_document_gold_is_unscored_not_zero():
    question = {**_question(), "expected_doc_ids": None, "gold_answer": None}
    result = _evaluator().evaluate(question, _prediction())
    assert result["loaded_document_recall"] is None
    assert result["reported_document_recall"] is None
    assert result["lexical_f1"] is None


def test_adapter_validates_each_metric_and_preserves_valid_denominators(monkeypatch):
    spec = _adapter(monkeypatch, lambda question, record: {
        "correctness": 90,
        "completeness": float("nan"),
        "overall": 120,
        "invalid_documents": 0.25,
    })
    record = _prediction()
    record["metrics"] = _evaluator(spec).evaluate(_question(), record)
    summary = aggregate([record])
    assert record["metrics"]["evaluation_status"] == "partial"
    assert summary["metrics"]["correctness"]["valid_count"] == 1
    assert summary["metrics"]["completeness"] == {"value": None, "valid_count": 0, "missing_or_invalid_count": 1}
    assert summary["metrics"]["overall"]["value"] is None
    assert summary["evaluation_failure_count"] == 1


def test_adapter_exception_never_becomes_zero_score(monkeypatch):
    def broken(question, record):
        raise RuntimeError("scorer unavailable")

    spec = _adapter(monkeypatch, broken)
    result = _evaluator(spec).evaluate(_question(), _prediction())
    assert result["evaluation_status"] == "failed"
    assert result["correctness"] is None
    assert "RuntimeError" in result["evaluation_errors"][0]


def test_runtime_failure_skips_adapter_and_answer_proxy(monkeypatch):
    calls = []
    spec = _adapter(monkeypatch, lambda *args: calls.append(args))
    result = _evaluator(spec).evaluate(_question(), _prediction(status="answer_failed", prediction="", trace={"loaded_chunk_ids": ["chunk-a", "chunk-b"]}))
    assert calls == []
    assert result["evaluation_status"] == "runtime_failed"
    assert result["correctness"] is None
    assert result["lexical_f1"] is None
    assert result["loaded_document_recall"] == 1.0


def test_adapter_cannot_mutate_frozen_prediction_or_evidence(monkeypatch):
    record = _prediction(evidence_chunks=[{"text": "two signatures"}])
    question = _question()

    def mutate(question_copy, record_copy):
        record_copy["prediction"] = "changed"
        record_copy["evidence_chunks"][0]["text"] = "changed"
        question_copy["gold_answer"] = "changed"
        return {"correctness": 100, "completeness": 100, "overall": 100, "invalid_documents": 0}

    result = _evaluator(_adapter(monkeypatch, mutate)).evaluate(question, record)
    assert result["evaluation_status"] == "scored"
    assert record["prediction"] == "two signatures"
    assert record["evidence_chunks"][0]["text"] == "two signatures"
    assert question["gold_answer"] == "two signatures"


def test_mismatched_question_ids_are_not_evaluated(monkeypatch):
    calls = []
    spec = _adapter(monkeypatch, lambda *args: calls.append(args))
    result = _evaluator(spec).evaluate(_question(), _prediction(question_id="wrong"))
    assert result["evaluation_status"] == "failed"
    assert calls == []


def test_adapter_preflight_rejects_noncallable_and_config_disallows_missing_adapter(monkeypatch):
    spec = _adapter(monkeypatch, 7)
    with pytest.raises(TypeError, match="not callable"):
        resolve_adapter(spec)
    with pytest.raises(ValueError, match="adapter is required"):
        Evaluator({"evaluation": {"adapter": None, "allow_unscored": False}})


def test_aggregate_reports_by_type_token_timing_and_valid_counts():
    first = _prediction(timing={"answer_seconds": 1.0}, usage={"answer": {"input_tokens": 100}}, measurements={"evidence_tokens": 80})
    first["metrics"] = _evaluator().evaluate(_question(), first)
    second = _prediction(status="failed", prediction="", reported_doc_ids=[], timing={"answer_seconds": 3.0}, usage={"answer": {"input_tokens": 300}})
    second["metrics"] = _evaluator().evaluate({**_question(), "question_type": "reasoning"}, second)
    summary = aggregate([first, second])
    assert summary["count"] == 2
    assert summary["metrics"]["lexical_f1"]["valid_count"] == 1
    assert summary["metrics"]["overall"]["valid_count"] == 0
    assert summary["timing"]["answer_seconds"]["mean"] == 2.0
    assert summary["timing"]["answer_seconds"]["p50"] == 2.0
    assert math.isclose(summary["timing"]["answer_seconds"]["p95"], 2.9)
    assert summary["usage"]["answer.input_tokens"]["mean"] == 200
    assert summary["evidence"]["evidence_tokens"]["mean"] == 80
    assert summary["evidence"]["evidence_tokens"]["valid_count"] == 1
    assert summary["reported_documents"]["mean"] == 0.5
    assert summary["runtime_failure_count"] == 1
    assert set(summary["by_type"]) == {"fact", "reasoning"}


def test_empty_aggregate_and_mixed_scales():
    assert aggregate([])["metrics"]["correctness"]["value"] is None
    first = _prediction(metrics={"score_scale": 100})
    second = _prediction(metrics={"score_scale": 1})
    with pytest.raises(ValueError, match="mixed score scales"):
        aggregate([first, second])


def test_failed_retrieval_does_not_create_valid_zero_document_recall():
    result = _evaluator().evaluate(_question(), _prediction(status="error", loaded_doc_ids=[], reported_doc_ids=[], trace={}))
    assert result["loaded_document_recall"] is None
    assert result["reported_document_recall"] is None
    assert result["evaluation_status"] == "runtime_failed"


def test_failed_answer_preserves_observed_empty_retrieval_denominator():
    result = _evaluator().evaluate(_question(), _prediction(status="error", loaded_doc_ids=[], reported_doc_ids=[], trace={"evidence_tokens": 0}))
    assert result["loaded_document_recall"] == 0.0
    assert result["reported_document_recall"] == 0.0
