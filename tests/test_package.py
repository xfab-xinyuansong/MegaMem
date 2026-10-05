from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import megamem
from megamem.cli import main
from megamem.document_eval import runner
from megamem.document_eval.runner import (
    ABSTENTION,
    SUITES,
    Evaluator,
    PaperRuntime,
    PersistentIndex,
    RunSpec,
    aggregate,
    build_identity,
    chunk_document,
    evidence_card,
    load_config,
    load_questions,
    load_records,
    make_specs,
    pack_evidence,
    resolve_adapter,
    resolve_and_fuse,
    resolved_config,
    select_questions,
    validate_abstraction,
    validate_atomic,
)


def test_version_is_exposed_without_loading_optional_backends() -> None:
    assert megamem.__version__ == "0.1.0"


def test_cli_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["megamem", "--version"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_module_entrypoint_exposes_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "megamem", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert megamem.__version__ in completed.stdout


def test_config_command_reports_active_file() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "megamem", "config"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Model config:" in completed.stdout
    assert "chat_low:" in completed.stdout
    assert "PLACEHOLDER" in completed.stdout


def test_doctor_json_is_safe_and_machine_readable(capsys) -> None:
    exit_code = main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["package"] == "MegaMem"
    assert report["core_ready"] is True
    assert report["models_ready"] is False
    assert set(report["optional_groups"]) == {
        "dev",
        "documents",
        "evaluation",
        "huggingface",
        "llm",
        "local-models",
        "retrieval",
    }
    assert "api_key" not in json.dumps(report).lower()


def test_package_root_exposes_lightweight_method_api() -> None:
    from megamem import (
        DualNode,
        GeneralAPIClient,
        TokenLedger,
        load_enterprise_rag_documents,
        validate_batch,
    )

    assert GeneralAPIClient.__name__ == "GeneralAPIClient"
    assert DualNode.__name__ == "DualNode"
    assert TokenLedger.__name__ == "TokenLedger"
    assert callable(load_enterprise_rag_documents)
    assert callable(validate_batch)


def test_core_install_does_not_import_optional_backends() -> None:
    script = """
import sys
from megamem import *
from megamem import MemoryClient
from megamem.methods import *
import megamem.utils.embedding

client = MemoryClient(api_key="test-token", server_url="https://memory.example")
assert client.is_remote
for module in ("chromadb", "httpx", "torch", "transformers"):
    assert module not in sys.modules, module
assert "megamem.core.local_client" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optional_interfaces_remain_discoverable() -> None:
    assert "DualIndex" in dir(megamem)
    assert "MemoryClient" in dir(megamem)


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


PAPER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.example.yaml"


def _config():
    config = load_config(PAPER_CONFIG)
    config["protocols"].update(
        no_dual_index_view="distilled",
        efficiency_extractor="full",
        detailed_evidence_budget=8192,
        detailed_max_documents=20,
    )
    return config


def _variants(config, suite):
    return {spec.variant: spec for spec in make_specs(config, [suite])}


def test_paper_configuration_leaves_unreported_choices_unset():
    config = load_config(PAPER_CONFIG)
    assert config["models"]["reranker"] is None
    assert config["evaluation"]["adapter"] is None
    for name in ("no_dual_index_view", "efficiency_extractor", "detailed_evidence_budget", "detailed_max_documents"):
        assert config["protocols"][name] is None
    assert config["runtime"] == {"seed": 42, "timeout_seconds": 120, "retries": 2, "temperature": 0.0}


def test_paper_configuration_distinguishes_parameters_and_implementation_choices():
    provenance = load_config(PAPER_CONFIG)["parameter_provenance"]
    assert set(provenance["experiment_suites"]) == set(SUITES)
    assert provenance["paper_given"]["split"]
    for name in ("retrieval.candidate_depth", "retrieval.rerank_candidates", "retrieval.route_weights", "retrieval.max_expansions", "index.abstraction_group_size"):
        assert provenance["implementation_defaults"][name]
    assert provenance["explicit_choices_required"]["evaluation.adapter"]
    assert provenance["experiment_suites"]["headline"]["unavailable_exact_historical_rows"]


def test_headline_uses_validation_at_10m_and_keeps_baselines_explicit():
    config = _config()
    variants = _variants(config, "headline")
    assert set(variants) == {"full", "no_query_expansion", "no_attribution", "bm25", "dense"}
    assert all((s.scale, s.extractor, s.split, s.expected_count) == ("10M", "full", "validation", 400) for s in variants.values())
    assert variants["no_attribution"].paired_from == "full"
    assert variants["no_query_expansion"].changes == {"query_expansion": False}
    for backend in ("bm25", "dense"):
        retrieval = resolved_config(config, variants[backend])["retrieval"]
        assert retrieval["backend"] == backend
        assert retrieval["raw_view"] is True
        assert all(retrieval[name] is False for name in ("distilled_view", "query_expansion", "rerank", "attribution"))


def test_scaling_uses_only_the_matched_mini_validation_trace():
    specs = make_specs(_config(), ["scaling"])
    assert [spec.scale for spec in specs] == ["20M", "60M", "100M", "150M", "250M"]
    assert all((s.extractor, s.split, s.expected_count, s.oracle) == ("mini", "validation", 400, False) for s in specs)


def test_gold_intervention_has_twelve_matched_and_explicit_oracle_runs():
    specs = make_specs(_config(), ["gold"])
    assert len(specs) == 12
    assert {(s.scale, s.extractor, s.variant) for s in specs} == {
        (scale, extractor, variant)
        for scale in ("20M", "60M", "250M")
        for extractor in ("mini", "full")
        for variant in ("retrieved", "gold")
    }
    for spec in specs:
        assert spec.split == "validation"
        assert spec.expected_count == 400
        assert spec.oracle is (spec.variant == "gold")
        assert spec.changes.get("evidence_source", "retrieved") == spec.variant


@pytest.mark.parametrize("view", [None, "both", "merged", ""])
def test_no_dual_index_requires_an_explicit_supported_view(view):
    config = _config()
    config["protocols"]["no_dual_index_view"] = view
    with pytest.raises(ValueError, match="no_dual_index_view"):
        make_specs(config, ["ablation"])


def test_distilled_single_view_ablation_is_distinct_from_no_distillation():
    config = _config()
    variants = _variants(config, "ablation")
    assert len(variants) == 6
    assert all((s.scale, s.extractor, s.split, s.expected_count) == ("10M", "full", "all", 500) for s in variants.values())
    no_distillation = resolved_config(config, variants["no_distillation"])
    no_dual = resolved_config(config, variants["no_dual_index"])
    assert (no_distillation["retrieval"]["raw_view"], no_distillation["retrieval"]["distilled_view"]) == (True, False)
    assert (no_dual["retrieval"]["raw_view"], no_dual["retrieval"]["distilled_view"]) == (False, True)
    assert no_dual["protocol"]["selected_index_view"] == "distilled"
    assert no_dual["protocol"]["equivalent_to"] is None
    assert variants["no_attribution"].paired_from == "full"


def test_raw_single_view_ablation_records_equivalence_instead_of_hiding_it():
    config = _config()
    config["protocols"]["no_dual_index_view"] = "raw"
    variants = _variants(config, "ablation")
    no_dual = resolved_config(config, variants["no_dual_index"])
    no_distillation = resolved_config(config, variants["no_distillation"])
    assert no_dual["retrieval"] == no_distillation["retrieval"]
    assert no_dual["protocol"]["equivalent_to"] == "no_distillation"


@pytest.mark.parametrize("extractor", [None, "unknown", ""])
def test_efficiency_requires_an_explicit_extractor(extractor):
    config = _config()
    config["protocols"]["efficiency_extractor"] = extractor
    with pytest.raises(ValueError, match="efficiency_extractor"):
        make_specs(config, ["evidence_efficiency"])


@pytest.mark.parametrize("name,value", [
    ("detailed_evidence_budget", None),
    ("detailed_evidence_budget", 4096),
    ("detailed_evidence_budget", True),
    ("detailed_max_documents", None),
    ("detailed_max_documents", 0),
    ("detailed_max_documents", True),
])
def test_full_detail_requires_explicit_valid_budget_and_document_limit(name, value):
    config = _config()
    config["protocols"][name] = value
    with pytest.raises(ValueError, match="full-detail|detailed_evidence_budget"):
        make_specs(config, ["evidence_efficiency"])


@pytest.mark.parametrize("extractor", ["mini", "full"])
def test_efficiency_matrix_preserves_the_chosen_extractor_and_paired_sample(extractor):
    config = _config()
    config["protocols"]["efficiency_extractor"] = extractor
    specs = make_specs(config, ["evidence_efficiency"])
    assert len(specs) == 6
    assert {(s.scale, s.variant) for s in specs} == {
        (scale, policy)
        for scale in ("10M", "20M")
        for policy in ("selective_detail", "detailed_only", "distilled_only")
    }
    for spec in specs:
        assert (spec.split, spec.expected_count, spec.sample_size, spec.extractor) == ("all", 100, 100, extractor)
        retrieval = resolved_config(config, spec)["retrieval"]
        assert retrieval["candidate_depth"] == 20
        assert retrieval["answer_tokens"] == 800
        assert retrieval["rrf_constant"] == 60
        expected = (8192, 20) if spec.variant == "detailed_only" else (4096, 5)
        assert (retrieval["evidence_budget"], retrieval["max_documents"]) == expected


@pytest.mark.parametrize("variant", ["selective_detail", "distilled_only"])
@pytest.mark.parametrize("key,value", [("evidence_budget", 8192), ("max_documents", 20)])
def test_efficiency_only_full_detail_can_relax_the_evidence_limits(variant, key, value):
    config = _config()
    config["retrieval"][key] = value
    spec = next(s for s in make_specs(config, ["evidence_efficiency"]) if s.variant == variant)
    with pytest.raises(ValueError, match=key):
        resolved_config(config, spec)


@pytest.mark.parametrize("key,value", [("answer_tokens", 1600), ("rrf_constant", 30)])
def test_even_full_detail_keeps_the_common_generation_and_fusion_settings(key, value):
    config = _config()
    config["retrieval"][key] = value
    spec = next(s for s in make_specs(config, ["evidence_efficiency"]) if s.variant == "detailed_only")
    with pytest.raises(ValueError, match=key):
        resolved_config(config, spec)


def test_diagnostics_label_oracle_and_preserve_attribution_pairing():
    variants = _variants(_config(), "diagnostics")
    assert set(variants) == {"unconditional", "no_expansion", "no_attribution", "selective_oracle"}
    assert all((s.scale, s.extractor, s.split, s.expected_count) == ("10M", "full", "all", 500) for s in variants.values())
    assert variants["selective_oracle"].oracle is True
    assert variants["selective_oracle"].paired_from == "unconditional"
    assert variants["no_attribution"].paired_from == "unconditional"
    assert variants["no_expansion"].changes == {"query_expansion": False}
    assert all(not s.oracle for name, s in variants.items() if name != "selective_oracle")


def test_transfer_preserves_counts_and_question_local_corpus_scope():
    specs = make_specs(_config(), ["transfer"])
    assert {(s.dataset, s.expected_count, s.corpus_scope) for s in specs} == {
        ("financebench", 150, "question_local"),
        ("hotpotqa", 200, "question_local"),
        ("locomo", 200, "shared"),
        ("ultradomain", 200, "shared"),
    }
    assert all((s.scale, s.extractor, s.split) == ("task", "full", "task") for s in specs)


def test_paths_resolve_relative_to_yaml_and_isolate_scale_and_extractor():
    config = _config()
    repo = PAPER_CONFIG.parents[2]
    for spec in make_specs(config, ["scaling", "gold"]):
        resolved = resolved_config(config, spec)
        assert Path(resolved["paths"]["questions"]) == repo / "data" / "enterprise" / "questions.jsonl"
        assert Path(resolved["paths"]["split_manifest"]) == repo / "data" / "enterprise" / "split_manifest.json"
        assert Path(resolved["paths"]["documents_template"]) == repo / "data" / "enterprise" / spec.scale / "documents.jsonl"
        assert Path(resolved["index"]["path"]) == repo / "data" / "indexes" / "enterprise" / spec.scale / spec.extractor
        assert Path(resolved["paths"]["output_root"]) == repo / "outputs" / "paper_experiments"
        assert resolved["index"]["scale_tokens"] == int(spec.scale[:-1]) * 1_000_000


def test_transfer_paths_do_not_reuse_enterprise_questions_or_split():
    config = _config()
    repo = PAPER_CONFIG.parents[2]
    for spec in make_specs(config, ["transfer"]):
        resolved = resolved_config(config, spec)
        assert Path(resolved["paths"]["questions"]) == repo / "data" / spec.dataset / "questions.jsonl"
        assert Path(resolved["paths"]["documents_template"]) == repo / "data" / spec.dataset / "documents.jsonl"
        assert Path(resolved["index"]["path"]) == repo / "data" / "indexes" / spec.dataset / "task" / "full"
        assert "split_manifest" not in resolved["paths"]
        assert "gold_evidence" not in resolved["paths"]
        assert resolved["index"]["scale_tokens"] is None


def test_run_ids_are_unique_and_duplicate_suite_requests_do_not_duplicate_runs():
    config = _config()
    specs = make_specs(config, list(SUITES))
    assert len(specs) == 42
    assert len({s.run_id for s in specs}) == len(specs)
    duplicates = make_specs(config, ["scaling", "headline", "scaling"])
    assert duplicates == make_specs(config, ["scaling", "headline"])


def test_resolving_runs_does_not_mutate_shared_configuration():
    config = _config()
    before = copy.deepcopy(config)
    for spec in make_specs(config, list(SUITES)):
        resolved = resolved_config(config, spec)
        assert resolved["retrieval"]["answer_tokens"] == 800
        assert resolved["retrieval"]["rrf_constant"] == 60
        assert resolved["models"]["embedding"] == "text-embedding-3-small"
    assert config == before


def test_unknown_suite_and_missing_enterprise_paths_fail_explicitly():
    config = _config()
    with pytest.raises(ValueError, match="Unknown suites"):
        make_specs(config, ["one_billion_tokens"])
    spec = make_specs(config, ["headline"])[0]
    config["paths"]["split_manifest"] = ""
    with pytest.raises(ValueError, match="split_manifest"):
        resolved_config(config, spec)


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


class CharacterCounter:
    def count(self, text):
        return len(text)


def configuration(tmp_path, **retrieval):
    return {"index": {"path": str(tmp_path)}, "retrieval": {"rerank": False, **retrieval}}


def raw_chunk(identifier, doc_id, content):
    return {
        "chunk_id": identifier, "doc_id": doc_id, "content": content,
        "section_path": "Section", "representation": "raw", "source_chunk_ids": [identifier],
    }


def test_runtime_construction_is_lazy_and_build_does_not_require_reranker(tmp_path):
    runtime = PaperRuntime({"index": {"path": str(tmp_path / "does-not-exist")}})
    assert runtime._store is None
    assert runtime._reranker is None
    assert runtime._clients == {}
    assert runtime.tokens._encoding is None
    assert not (tmp_path / "does-not-exist").exists()


def test_resolve_precedes_deduplication_and_reciprocal_rank_fusion():
    routes = [
        {"route_id": "original:distilled", "query": "q", "view": "distilled_memory", "weight": 1.0,
         "hits": [
             {"id": "m1", "source_chunk_ids": ["a"]},
             {"id": "m2", "source_chunk_ids": ["a"]},
             {"id": "m3", "source_chunk_ids": ["b"]},
         ]},
        {"route_id": "original:raw", "query": "q", "view": "raw_chunks", "weight": 1.0,
         "hits": [{"id": "b", "source_chunk_ids": ["b"]}]},
    ]
    ranking, scores, trace = resolve_and_fuse(routes, 60)
    assert ranking == ["b", "a"]
    assert scores["a"]["rrf_score"] == pytest.approx(1 / 61)
    assert scores["b"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert trace[0]["resolved_chunk_ids"] == ["a", "b"]


def test_multi_source_abstraction_resolves_every_child_before_fusion():
    routes = [{
        "route_id": "canonical", "query": "q", "view": "distilled_memory", "weight": 0.5,
        "hits": [{"id": "group", "source_chunk_ids": ["b", "a"]}],
    }]
    ranking, scores, _ = resolve_and_fuse(routes, 60)
    assert ranking == ["b", "a"]
    assert scores["a"]["rrf_score"] == pytest.approx(0.5 / 62)


def test_packing_counts_labels_and_separators_and_respects_document_limit():
    chunks = [raw_chunk("a", "doc-a", "one"), raw_chunk("b", "doc-b", "two")]
    tokens = CharacterCounter()
    budget = tokens.count(evidence_card(chunks[0]))
    selected, text, skipped = pack_evidence(chunks, tokens, budget, 5)
    assert [chunk["chunk_id"] for chunk in selected] == ["a"]
    assert tokens.count(text) == budget
    assert skipped == [{"chunk_id": "b", "reason": "evidence_token_budget"}]
    selected, _, skipped = pack_evidence(chunks, tokens, 10000, 1)
    assert [chunk["doc_id"] for chunk in selected] == ["doc-a"]
    assert skipped[0]["reason"] == "document_budget"


def test_section_chunks_are_exact_source_substrings_and_bounded():
    source = "# One\n" + "alpha beta " * 15 + "\n## Two\n" + "gamma " * 20
    document = {"doc_id": "d", "content": source}
    chunks = chunk_document(document, CharacterCounter(), 70)
    assert any(chunk["section_path"] == "One / Two" for chunk in chunks)
    assert all(chunk["token_count"] <= 70 for chunk in chunks)
    assert all(chunk["content"] == source[chunk["start_char"]:chunk["end_char"]] for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_typed_memory_limit_and_abstraction_source_identity_are_strict():
    valid = {"type": "fact", "key": "term", "value": "statement"}
    with pytest.raises(ValueError, match="three"):
        validate_atomic({"memories": [valid] * 4})
    with pytest.raises(ValueError, match="identifiers"):
        validate_abstraction({"search_key": "term", "summary": "statement", "source_chunk_ids": ["wrong"]}, ["a"])


def test_gold_answer_is_never_forwarded_and_attribution_cannot_change_prediction(tmp_path):
    runtime = PaperRuntime(configuration(tmp_path, evidence_source="gold", attribution=True))
    runtime.tokens = CharacterCounter()
    prompts = []

    def chat(stage, system, user, limit, json_output=False):
        prompts.append((stage, user))
        return "Evidence-supported answer."

    def attribute(stage, system, user, validator, fallback, limit):
        prompts.append((stage, user))
        return ["doc-a", "unloaded-doc"]

    runtime._chat = chat
    runtime._json_call = attribute
    result = runtime.answer({
        "question_id": "q", "question": "What does the evidence say?", "gold_answer": "SECRET GOLD ANSWER",
        "gold_chunks": [raw_chunk("a", "doc-a", "Only this original evidence may be used.")],
    })
    assert result["prediction"] == "Evidence-supported answer."
    assert result["reported_doc_ids"] == ["doc-a"]
    assert result["trace"]["attribution_rejected_doc_ids"] == ["unloaded-doc"]
    assert all("SECRET GOLD ANSWER" not in prompt for _, prompt in prompts)
    assert [stage for stage, _ in prompts] == ["answer", "attribution"]
    assert runtime._store is None


def test_explicit_empty_gold_evidence_abstains_without_api(tmp_path):
    runtime = PaperRuntime(configuration(tmp_path, evidence_source="gold"))
    runtime.tokens = CharacterCounter()
    result = runtime.answer({"question_id": "q", "question": "Unanswerable?", "gold_chunks": []})
    assert result["status"] == "ok"
    assert result["prediction"] == ABSTENTION
    assert result["loaded_doc_ids"] == []
    assert runtime._clients == {}


def test_incomplete_index_is_rejected_before_any_api(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    runtime = PaperRuntime(configuration(tmp_path))
    result = runtime.answer({"question_id": "q", "question": "Question"})
    assert result["status"] == "error"
    assert "completed" in result["errors"][0]["message"]
    assert runtime._clients == {}


def test_inference_collection_uses_get_and_preserves_scope(tmp_path):
    calls = []

    class Collection:
        metadata = {"hnsw:space": "cosine"}

        def count(self):
            return 10

        def query(self, **kwargs):
            calls.append(kwargs)
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

    def get_collection(**kwargs):
        calls.append(kwargs)
        return Collection()

    index = PersistentIndex({"index": {"path": str(tmp_path)}})
    index._chroma = SimpleNamespace(get_collection=get_collection)
    assert index.query("raw_chunks", [0.1], 5, ["local-doc"]) == []
    assert calls[0] == {"name": "raw_chunks", "embedding_function": None}
    assert calls[1]["where"] == {"doc_id": {"$in": ["local-doc"]}}


def test_bm25_does_not_leak_outside_question_corpus(tmp_path):
    index = PersistentIndex({"index": {"path": str(tmp_path)}}, build_mode=True)
    try:
        for position, doc_id in enumerate(["local", "other"]):
            index.stage_document(
                {"doc_id": doc_id, "fingerprint": doc_id, "position": position, "token_count": 2},
                [raw_chunk(doc_id + "-chunk", doc_id, "alpha beta")], [],
            )
            index.finish_document(doc_id)
        hits = index.bm25("alpha", 20, 1.5, 0.75, ["local"])
        assert [hit["doc_id"] for hit in hits] == ["local"]
        assert index.bm25("alpha", 20, 1.5, 0.75, []) == []
    finally:
        index.close()


def test_build_identity_is_extractor_isolated_but_retrieval_independent(tmp_path):
    first = PaperRuntime(configuration(tmp_path))
    second = PaperRuntime(configuration(tmp_path, query_expansion=False, attribution=False))
    assert build_identity(first.config) == build_identity(second.config)
    second.config["index"]["extractor"] = "full"
    assert build_identity(first.config) != build_identity(second.config)
