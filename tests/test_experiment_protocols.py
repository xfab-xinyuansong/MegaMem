from __future__ import annotations

import copy
from pathlib import Path

import pytest

from megamem.experiments.config import load_config
from megamem.experiments.protocols import SUITES, make_specs, resolved_config


PAPER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "paper.yaml"


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
