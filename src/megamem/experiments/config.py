from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULTS = {
    "models": {
        "extraction": "gpt-5.4-mini",
        "abstraction": "gpt-5.4",
        "query": "gpt-5.4",
        "answer": "gpt-5.4",
        "attribution": "gpt-5.4",
        "judge": "gpt-5.4",
        "embedding": "text-embedding-3-small",
        "reranker": None,
    },
    "retrieval": {
        "candidate_depth": 20,
        "rerank_candidates": 100,
        "max_documents": 5,
        "evidence_budget": 4096,
        "answer_tokens": 800,
        "rrf_constant": 60,
        "original_weight": 1.0,
        "canonical_weight": 1.0,
        "expansion_weight": 0.5,
        "max_expansions": 3,
        "raw_view": True,
        "distilled_view": True,
        "query_expansion": True,
        "rerank": True,
        "attribution": True,
        "evidence_policy": "selective_detail",
        "backend": "dual",
        "evidence_source": "retrieved",
    },
    "runtime": {
        "seed": 42,
        "timeout_seconds": 120,
        "retries": 2,
        "temperature": 0.0,
    },
    "index": {
        "chunk_tokens": 400,
        "chunk_overlap": 0,
        "abstraction_group_size": 12,
        "embedding_dimensions": 1536,
        "encoding": "cl100k_base",
        "enforce_token_cap": False,
    },
    "evaluation": {"adapter": None, "score_scale": 100, "allow_unscored": True},
    "protocols": {
        "no_dual_index_view": None,
        "efficiency_extractor": None,
        "detailed_evidence_budget": None,
        "detailed_max_documents": None,
    },
    "paths": {},
    "datasets": {},
}


def merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def public_config(config: dict) -> dict:
    forbidden = {"api_key", "access_token", "auth_token", "password", "secret", "authorization"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: "<redacted>" if any(word in k.lower() for word in forbidden) or k.lower() == "token" else scrub(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return scrub(config)


def load_config(path: str | Path) -> dict:
    import yaml

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration must be a mapping")
    unknown = set(raw) - set(DEFAULTS) - {"parameter_provenance"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for section in ("models", "retrieval", "runtime", "index", "evaluation", "protocols", "paths", "datasets"):
        if section in raw and not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping")
    config = merge(DEFAULTS, raw)
    config["config_path"] = str(source)
    config["config_directory"] = str(source.parent)
    if any(config["runtime"].get(k) for k in ("api_key", "embedding_api_key")):
        raise ValueError("Use environment variables for API credentials")
    for section, keys in {
        "retrieval": ("candidate_depth", "rerank_candidates", "max_documents", "evidence_budget", "answer_tokens", "rrf_constant"),
        "index": ("chunk_tokens", "abstraction_group_size", "embedding_dimensions"),
    }.items():
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("original_weight", "canonical_weight", "expansion_weight"):
        value = config["retrieval"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"retrieval.{key} must be finite and positive")
    for key in ("raw_view", "distilled_view", "query_expansion", "rerank", "attribution"):
        if not isinstance(config["retrieval"][key], bool):
            raise ValueError(f"retrieval.{key} must be boolean")
    if config["runtime"]["seed"] != 42 or config["runtime"]["temperature"] != 0:
        raise ValueError("Paper protocols require seed=42 and temperature=0")
    if config["runtime"]["timeout_seconds"] != 120:
        raise ValueError("Paper protocols require timeout_seconds=120")
    retries = config["runtime"]["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries not in {0, 1, 2}:
        raise ValueError("runtime.retries must be an integer between zero and two")
    expansions = config["retrieval"]["max_expansions"]
    if isinstance(expansions, bool) or not isinstance(expansions, int) or expansions < 0:
        raise ValueError("retrieval.max_expansions must be a nonnegative integer")
    overlap = config["index"]["chunk_overlap"]
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("index.chunk_overlap must be an integer")
    if config["index"]["chunk_overlap"] < 0 or config["index"]["chunk_overlap"] >= config["index"]["chunk_tokens"]:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_tokens")
    return config


def path_for(config: dict, template: str, **values: Any) -> Path:
    path = Path(template.format(**values)).expanduser()
    if not path.is_absolute():
        path = Path(config["config_directory"]) / path
    return path.resolve()
