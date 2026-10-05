from __future__ import annotations

import copy
import json

import pytest

from megamem.experiments import runner
from megamem.experiments.evaluation import Evaluator
from megamem.experiments.protocols import RunSpec


def _source_record():
    return {
        "question_id": "q1",
        "prediction": "A fixed answer.",
        "loaded_doc_ids": ["a", "b"],
        "retrieved_doc_ids": ["a", "b", "c"],
        "reported_doc_ids": ["a"],
        "evidence_chunks": [{"chunk_id": "c1", "doc_id": "a", "text": "Evidence"}],
        "status": "ok",
        "timing": {"total_seconds": 2},
        "usage": {"total_tokens": 100},
        "trace": {"reported_doc_ids": ["a"], "loaded_chunk_ids": ["c1"], "evidence_tokens": 80},
        "metrics": {
            "correctness": 90.0,
            "completeness": 80.0,
            "overall": 85.0,
            "invalid_documents": 0.1,
            "lexical_f1": 0.8,
            "loaded_document_recall": 1.0,
            "reported_document_recall": 0.5,
            "evaluation_status": "scored",
            "evaluation_errors": [],
            "score_scale": 100,
        },
    }


def _manifest(spec, scale=100, adapter="official:score"):
    return {
        "fingerprint": "run-fingerprint",
        "identity": {
            "question_ids": ["q1"],
            "questions_digest": "full-question-content",
            "comparison_questions_digest": "shared-gold-and-question-content",
            "evaluation_adapter_digest": "adapter-source-digest",
            "configuration": {
                "protocol": spec.to_dict(),
                "evaluation": {"score_scale": scale, "adapter": adapter, "adapter_version": "v1"},
            },
        },
    }


def _spec(variant="full", paired_from=None):
    return RunSpec("ablation", "enterprise", "10M", "full", "all", 1, variant, {}, paired_from=paired_from)


def _write_run(path, spec=None, scale=100, adapter="official:score", record=None):
    spec = spec or _spec()
    path.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(spec, scale, adapter)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = copy.deepcopy(record if record is not None else _source_record())
    row["run_fingerprint"] = manifest["fingerprint"]
    target = runner.record_path(path, "q1")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(row), encoding="utf-8")
    return manifest


def test_runtime_question_excludes_all_gold_and_oracle_labels():
    question = {
        "question_id": "q1", "question": "Question?", "corpus_doc_ids": ["a"],
        "gold_answer": "Secret gold", "expected_doc_ids": ["a"], "answer_facts": ["Fact"],
        "gold_chunks": [{"text": "Gold evidence"}], "answerable": True,
    }
    visible = runner.runtime_question(question)
    assert set(visible) == {"question_id", "question", "corpus_doc_ids"}
    visible["corpus_doc_ids"].append("b")
    assert question["corpus_doc_ids"] == ["a"]


def test_gold_intervention_only_adds_explicit_evidence():
    question = {"question_id": "q1", "question": "Question?", "gold_answer": "Secret gold", "gold_chunks": [{"text": "Gold evidence"}]}
    visible = runner.runtime_question(question, gold=True)
    assert set(visible) == {"question_id", "question", "gold_chunks"}
    visible["gold_chunks"][0]["text"] = "Changed"
    assert question["gold_chunks"][0]["text"] == "Gold evidence"
    with pytest.raises(ValueError, match="Missing explicit gold_chunks"):
        runner.runtime_question({"question_id": "q1", "question": "Question?"}, gold=True)


def test_record_filename_cannot_escape_records_directory(tmp_path):
    destination = runner.record_path(tmp_path, "../../outside.json")
    assert destination.parent == tmp_path / "records"
    assert destination.suffix == ".json"
    assert len(destination.stem) == 64
    assert destination == runner.record_path(tmp_path, "../../outside.json")
    assert destination != runner.record_path(tmp_path, "another-id")


def test_no_attribution_pair_preserves_answer_evidence_and_source_record():
    source = _source_record()
    original = copy.deepcopy(source)
    paired = runner.paired_record(source, "no_attribution", "source-run")
    assert source == original
    assert paired["prediction"] == source["prediction"]
    assert paired["evidence_chunks"] == source["evidence_chunks"]
    assert paired["loaded_doc_ids"] == source["loaded_doc_ids"]
    assert paired["reported_doc_ids"] == ["a", "b"]
    assert paired["trace"]["reported_doc_ids"] == ["a", "b"]
    assert paired["pairing"]["additional_generation_calls"] == 0
    assert paired["timing"] == paired["usage"] == {}
    assert paired["paired_source_execution"]["usage"] == source["usage"]
    paired["evidence_chunks"][0]["text"] = "Changed"
    assert source == original


def test_paired_attribution_metrics_reuse_answer_scores():
    source = _source_record()
    paired = runner.paired_record(source, "no_attribution", "source-run")

    class FreshEvaluator:
        def evaluate(self, question, record):
            return {**source["metrics"], "correctness": 1, "completeness": 2, "overall": 3,
                    "lexical_f1": 0.1, "loaded_document_recall": 0.2,
                    "reported_document_recall": 1.0, "invalid_documents": 0.3}

    metrics = runner.paired_metrics({"question_id": "q1"}, paired, source["metrics"], FreshEvaluator())
    for key in ("correctness", "completeness", "overall", "lexical_f1", "loaded_document_recall"):
        assert metrics[key] == source["metrics"][key]
    assert metrics["reported_document_recall"] == 1.0
    assert metrics["invalid_documents"] == 0.3


def test_selective_oracle_reuses_metrics_without_judge_call():
    source = _source_record()
    paired = runner.paired_record(source, "selective_oracle", "selected-source-run")

    class ForbiddenEvaluator:
        def evaluate(self, question, record):
            raise AssertionError("Oracle pairing must not call the judge again")

    metrics = runner.paired_metrics({"question_id": "q1"}, paired, source["metrics"], ForbiddenEvaluator())
    assert metrics == source["metrics"]
    metrics["evaluation_errors"].append("Changed")
    assert source["metrics"]["evaluation_errors"] == []


def test_resume_reuses_saved_record_without_generation_or_evaluation(tmp_path):
    spec = _spec()
    manifest = _write_run(tmp_path, spec)

    class ForbiddenRuntime:
        def answer(self, question):
            raise AssertionError("Saved answers must not be regenerated")

    class ForbiddenEvaluator:
        def evaluate(self, question, record):
            raise AssertionError("Saved metrics must not be rejudged")

    item = {"spec": spec, "directory": tmp_path, "manifest": manifest,
            "questions": [{"question_id": "q1", "question": "Question?"}],
            "runtime": ForbiddenRuntime(), "evaluator": ForbiddenEvaluator()}
    records = runner._execute_run(item, {})
    assert records["q1"]["prediction"] == "A fixed answer."


def test_resume_rejects_record_from_another_fingerprint(tmp_path):
    spec = _spec()
    manifest = _write_run(tmp_path, spec)
    path = runner.record_path(tmp_path, "q1")
    saved = json.loads(path.read_text())
    saved["run_fingerprint"] = "different-run"
    path.write_text(json.dumps(saved), encoding="utf-8")
    item = {"spec": spec, "directory": tmp_path, "manifest": manifest,
            "questions": [{"question_id": "q1", "question": "Question?"}]}
    with pytest.raises(ValueError, match="mismatched record"):
        runner._execute_run(item, {})


def test_runtime_failure_is_saved_with_null_scores(tmp_path):
    spec = _spec()

    class FailingRuntime:
        def answer(self, question):
            assert set(question) == {"question_id", "question"}
            raise RuntimeError("Unavailable")

    item = {"spec": spec, "directory": tmp_path, "manifest": _manifest(spec),
            "questions": [{"question_id": "q1", "question": "Question?", "gold_answer": "Gold"}],
            "runtime": FailingRuntime(), "evaluator": Evaluator({}),
            "config": {"retrieval": {"evidence_source": "retrieved"}, "protocol": spec.to_dict()}}
    records = runner._execute_run(item, {})
    result = records["q1"]
    assert result["status"] == "failed"
    assert result["metrics"]["correctness"] is None
    assert result["metrics"]["loaded_document_recall"] is None
    assert runner.record_path(tmp_path, "q1").is_file()


@pytest.mark.parametrize("scale,adapter,match", [(1, "official:score", "score_scale"), (100, "different:score", "adapter")])
def test_compare_rejects_incompatible_evaluators(tmp_path, scale, adapter, match):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    _write_run(second, scale=scale, adapter=adapter)
    with pytest.raises(ValueError, match=match):
        runner.compare(first, second)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True])
def test_compare_excludes_nonfinite_and_boolean_metrics(tmp_path, bad_value):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    changed = _source_record()
    changed["metrics"]["correctness"] = bad_value
    _write_run(second, record=changed)
    result = runner.compare(first, second)
    assert result["metric_changes"]["correctness"]["paired_valid_count"] == 0
    assert result["metric_changes"]["correctness"]["mean_difference_second_minus_first"] is None


def test_compare_uses_matching_gold_and_question_content(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    manifest = _write_run(second)
    manifest["identity"]["comparison_questions_digest"] = "different-gold"
    (second / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison_questions_digest"):
        runner.compare(first, second)


def test_compare_keeps_null_official_metrics_out_of_denominators(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    changed = _source_record()
    changed["metrics"]["overall"] = None
    changed["reported_doc_ids"] = ["a", "b"]
    changed["metrics"]["reported_document_recall"] = 1.0
    _write_run(second, record=changed)
    result = runner.compare(first, second)
    assert result["metric_changes"]["overall"]["paired_valid_count"] == 0
    assert result["metric_changes"]["reported_document_recall"]["mean_difference_second_minus_first"] == 0.5
    assert result["identical_answers"] == 1
