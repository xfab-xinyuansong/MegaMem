from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from .config import merge, path_for


SUITES = ("headline", "scaling", "ablation", "gold", "evidence_efficiency", "diagnostics", "transfer")


@dataclass(frozen=True)
class RunSpec:
    suite: str
    dataset: str
    scale: str
    extractor: str
    split: str
    expected_count: int
    variant: str
    changes: dict
    sample_size: int | None = None
    paired_from: str | None = None
    oracle: bool = False
    corpus_scope: str = "shared"

    @property
    def run_id(self) -> str:
        return "__".join((self.suite, self.dataset, self.scale, self.extractor, self.split, self.variant))

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, **asdict(self)}


def make_specs(config: dict, suites: list[str]) -> list[RunSpec]:
    invalid = set(suites) - set(SUITES)
    if invalid:
        raise ValueError(f"Unknown suites: {sorted(invalid)}")
    specs: list[RunSpec] = []

    def add(suite, scale, extractor, split, count, variant="full", changes=None, **kwargs):
        specs.append(RunSpec(suite, "enterprise", scale, extractor, split, count, variant, changes or {}, **kwargs))

    for suite in dict.fromkeys(suites):
        if suite == "headline":
            add(suite, "10M", "full", "validation", 400)
            add(suite, "10M", "full", "validation", 400, "no_query_expansion", {"query_expansion": False})
            add(suite, "10M", "full", "validation", 400, "no_attribution", {"attribution": False}, paired_from="full")
            for backend in ("bm25", "dense"):
                add(suite, "10M", "full", "validation", 400, backend, {
                    "backend": backend, "raw_view": True, "distilled_view": False,
                    "query_expansion": False, "rerank": False, "attribution": False,
                })
        elif suite == "scaling":
            for scale in ("20M", "60M", "100M", "150M", "250M"):
                add(suite, scale, "mini", "validation", 400)
        elif suite == "ablation":
            view = config["protocols"].get("no_dual_index_view")
            if view not in {"raw", "distilled"}:
                raise ValueError("Set protocols.no_dual_index_view to raw or distilled; the manuscript does not define this intervention")
            add(suite, "10M", "full", "all", 500)
            for variant, changes in (
                ("no_query_expansion", {"query_expansion": False}),
                ("no_reranker", {"rerank": False}),
                ("no_distillation", {"raw_view": True, "distilled_view": False}),
                ("no_dual_index", {"raw_view": view == "raw", "distilled_view": view == "distilled"}),
            ):
                add(suite, "10M", "full", "all", 500, variant, changes)
            add(suite, "10M", "full", "all", 500, "no_attribution", {"attribution": False}, paired_from="full")
        elif suite == "gold":
            for extractor in ("mini", "full"):
                for scale in ("20M", "60M", "250M"):
                    add(suite, scale, extractor, "validation", 400, "retrieved")
                    add(suite, scale, extractor, "validation", 400, "gold", {"evidence_source": "gold"}, oracle=True)
        elif suite == "evidence_efficiency":
            extractor = config["protocols"].get("efficiency_extractor")
            if extractor not in {"mini", "full"}:
                raise ValueError("Set protocols.efficiency_extractor explicitly; the historical retained stack is not specified")
            budget = config["protocols"].get("detailed_evidence_budget")
            documents = config["protocols"].get("detailed_max_documents")
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (budget, documents)):
                raise ValueError("Set protocols.detailed_evidence_budget and detailed_max_documents explicitly for the full-detail diagnostic")
            if budget <= 4096:
                raise ValueError("The full-detail diagnostic requires an explicit budget greater than the 4096-token selective policy")
            for scale in ("10M", "20M"):
                for policy in ("selective_detail", "detailed_only", "distilled_only"):
                    changes = {"candidate_depth": 20, "evidence_policy": policy}
                    if policy == "detailed_only":
                        changes.update(evidence_budget=budget, max_documents=documents)
                    add(suite, scale, extractor, "all", 100, policy, changes, sample_size=100)
        elif suite == "diagnostics":
            add(suite, "10M", "full", "all", 500, "unconditional")
            add(suite, "10M", "full", "all", 500, "no_expansion", {"query_expansion": False})
            add(suite, "10M", "full", "all", 500, "no_attribution", {"attribution": False}, paired_from="unconditional")
            add(suite, "10M", "full", "all", 500, "selective_oracle", paired_from="unconditional", oracle=True)
        elif suite == "transfer":
            for dataset, count, scope in (("financebench", 150, "question_local"), ("hotpotqa", 200, "question_local"), ("locomo", 200, "shared"), ("ultradomain", 200, "shared")):
                specs.append(RunSpec(suite, dataset, "task", "full", "task", count, "full", {}, corpus_scope=scope))
    return specs


def resolved_config(config: dict, spec: RunSpec) -> dict:
    result = copy.deepcopy(config)
    defaults = {"raw_view": True, "distilled_view": True, "query_expansion": True, "rerank": True,
                "attribution": True, "backend": "dual", "evidence_policy": "selective_detail", "evidence_source": "retrieved"}
    result["retrieval"] = merge(merge(result["retrieval"], defaults), spec.changes)
    for key, expected in (("evidence_budget", 4096), ("answer_tokens", 800), ("max_documents", 5), ("rrf_constant", 60)):
        if spec.suite == "evidence_efficiency" and spec.variant == "detailed_only" and key in {"evidence_budget", "max_documents"}:
            continue
        if result["retrieval"][key] != expected:
            raise ValueError(f"Paper protocol requires retrieval.{key}={expected}")
    values = {"dataset": spec.dataset, "scale": spec.scale, "extractor": spec.extractor}
    dataset_paths = result["datasets"].get(spec.dataset, {})
    paths = merge(result["paths"], dataset_paths)
    required = ["questions", "documents_template", "index_template", "output_root"]
    if spec.dataset == "enterprise":
        required.append("split_manifest")
    for key in required:
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"Configure a non-empty path for {spec.dataset}.{key}")
    resolved = {key: str(path_for(result, paths[key], **values)) for key in required}
    if paths.get("gold_evidence_template"):
        resolved["gold_evidence"] = str(path_for(result, paths["gold_evidence_template"], **values))
    result["paths"] = resolved
    result["index"].update(
        path=resolved["index_template"],
        extractor=spec.extractor,
        scale_tokens=int(spec.scale[:-1]) * 1_000_000 if spec.scale.endswith("M") else None,
        embedding_model=result["models"]["embedding"],
    )
    result["protocol"] = spec.to_dict()
    if spec.variant == "no_dual_index":
        result["protocol"]["selected_index_view"] = config["protocols"]["no_dual_index_view"]
        result["protocol"]["equivalent_to"] = "no_distillation" if config["protocols"]["no_dual_index_view"] == "raw" else None
    return result
