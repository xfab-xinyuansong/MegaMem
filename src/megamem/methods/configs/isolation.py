"""Released experiment-protocol isolation checks.

Enforces hard configuration constraints at config-load time:

- No governance / RBAC / memory_access fields
- No LoCoMo / LoCoMo+ / subset / sample / max-N keys
- No silent low-tier fallback for high-tier alias

Any forbidden field must raise ``V4ConfigError`` (a retained compatibility name
and subclass of ``RuntimeError``) so it
propagates up to the runner and aborts the job with a clear message).

Usage:
    from megamem_v4.configs.isolation import validate_config
    validate_config(config_dict)   # raises on any violation
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import yaml


# -----------------------------------------------------------------------
# Constants (kept in sync with schema.yaml — copied here so checks don't
# silently bypass when schema.yaml is missing.)
# -----------------------------------------------------------------------

FORBIDDEN_KEYS = {
    "governance", "rbac", "memory_access", "access_label",
    "deterministic_access_comparison", "redaction_accuracy", "over_refusal",
    "allow_redact_refuse", "evermemos", "solution1",
    "max_docs", "max_queries", "n_evidence_max", "first_n",
    "subset", "limit_queries", "limit_docs",
    "sample_queries", "random_query_subsample", "skip_failed_queries",
}

FORBIDDEN_DATASETS = {
    "locomo", "locomo_plus", "locomo10",
    "own_rag_subset", "erag_subset",
}

ALLOWED_DATASETS = {
    "own_full", "erag_50m", "erag_100m", "erag_150m", "erag_250m",
}

ALLOWED_METHODS = {
    "B1", "B2", "B3", "B4", "B5", "V3-NG", "V4",
}


class V4ConfigError(RuntimeError):
    """Raised when the released experiment protocol is violated."""


# -----------------------------------------------------------------------
# Core validation
# -----------------------------------------------------------------------

def _walk_keys(d: Any, parent: str = "") -> List[tuple]:
    """Walk every key/path in a nested dict-or-list; yield (path, key, value)."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{parent}.{k}" if parent else str(k)
            out.append((path, k, v))
            out.extend(_walk_keys(v, path))
    elif isinstance(d, list):
        for i, item in enumerate(d):
            path = f"{parent}[{i}]"
            out.extend(_walk_keys(item, path))
    return out


def _normalize(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    # case-insensitive + space/hyphen normalized
    return re.sub(r"[\s_\-]+", "_", s.strip().lower())


def validate_config(config: Dict[str, Any]) -> None:
    """Validate an experiment config; raise on any protocol violation."""
    if not isinstance(config, dict):
        raise V4ConfigError(f"config must be a dict, got {type(config).__name__}")

    # 1. Check required top-level fields
    for required in ("run_id", "dataset", "method", "seed", "models", "paths", "tokenizer"):
        if required not in config:
            raise V4ConfigError(f"missing required field: {required!r}")

    # 2. Method check
    method = config.get("method")
    if method not in ALLOWED_METHODS:
        raise V4ConfigError(
            f"method={method!r} not in allowed set {sorted(ALLOWED_METHODS)}"
        )

    # 3. Dataset check
    dataset = _normalize(config.get("dataset", ""))
    if dataset in {_normalize(d) for d in FORBIDDEN_DATASETS}:
        raise V4ConfigError(
            f"dataset={config.get('dataset')!r} is FORBIDDEN by the released protocol. "
            "Use own_full or erag_{50,100,150,250}m only."
        )
    if dataset not in {_normalize(d) for d in ALLOWED_DATASETS}:
        raise V4ConfigError(
            f"dataset={config.get('dataset')!r} not in allowed set {sorted(ALLOWED_DATASETS)}"
        )

    # 4. Forbidden keys (anywhere in the config tree)
    violations = []
    for path, key, value in _walk_keys(config):
        key_norm = _normalize(key)
        if key_norm in {_normalize(f) for f in FORBIDDEN_KEYS}:
            violations.append((path, key, value))
    if violations:
        msg = "FORBIDDEN config keys detected by the released protocol:\n"
        for path, k, v in violations[:20]:
            sv = json.dumps(v)[:60] if not isinstance(v, (dict, list)) else f"<{type(v).__name__}>"
            msg += f"  {path}: {k}={sv}\n"
        raise V4ConfigError(msg)

    # 5. Forbidden dataset values inside config tree (substring match for paths,
    # IDs, embedded references)
    for path, key, value in _walk_keys(config):
        if isinstance(value, str):
            v_norm = _normalize(value)
            for f in FORBIDDEN_DATASETS:
                f_norm = _normalize(f)
                if f_norm == v_norm or (len(f_norm) >= 6 and f_norm in v_norm):
                    raise V4ConfigError(
                        f"FORBIDDEN dataset reference at {path}={value!r} (matches {f!r}). "
                        "Released main runs may not reference LoCoMo / LoCoMo+ / subset paths."
                    )

    # 6. Models block sanity
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise V4ConfigError("models must be a dict")
    for slot in ("hierarchy_low", "hierarchy_high", "answer", "judge"):
        if slot not in models:
            raise V4ConfigError(f"models is missing required slot: {slot!r}")
    # Fallback guard for high-tier slots.
    high = _normalize(str(models.get("hierarchy_high", "")))
    answer = _normalize(str(models.get("answer", "")))
    for label, val in (("hierarchy_high", high), ("answer", answer)):
        if val in {"chat_low", "low_tier", "small_model", "cheap_model"}:
            raise V4ConfigError(
                f""
                f"models.{label}={val!r} is a forbidden silent substitute for chat_high. "
                "Reroute through an explicit substitute alias with maintainer sign-off."
            )

    # 7. seed sanity
    seed = config.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise V4ConfigError(f"seed must be a non-negative int, got {seed!r}")

    # 8. promotion / decay only allowed on V4 / B5
    if "promotion" in config and method not in {"V4", "B5"}:
        raise V4ConfigError(
            f"promotion config block is only valid for V4 or B5, not {method!r}"
        )
    if "decay" in config and method != "V4":
        raise V4ConfigError(
            f"decay config block is only valid for V4, not {method!r}"
        )


# -----------------------------------------------------------------------
# Convenience loaders
# -----------------------------------------------------------------------

def load_and_validate(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config from path, validate it, return the dict."""
    if not os.path.exists(path):
        raise V4ConfigError(f"config path does not exist: {path}")
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        cfg = yaml.safe_load(text)
    elif path.endswith(".json"):
        cfg = json.loads(text)
    else:
        raise V4ConfigError(f"unsupported config extension: {path}")
    validate_config(cfg)
    return cfg


# -----------------------------------------------------------------------
# Self-test (called from step1 smoke runner)
# -----------------------------------------------------------------------

def _self_test() -> int:
    """Build a few synthetic configs and verify FORBIDDEN keys / datasets trip the guard.

    Returns 0 on full pass, 1 if any negative case slipped through.
    """
    cases = [
        # (label, config, should_fail, expected_error_substring)
        (
            "valid V4 own_full",
            dict(
                run_id="r1", dataset="own_full", method="V4", seed=42,
                models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                            answer="chat_high", judge="judge"),
                paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
                tokenizer="cl100k_base",
                promotion=dict(score_weights={}, decision_mode="llm", promotion_budget=100),
                decay=dict(tau=20, decay_window=50, utility_floor=0.1),
            ),
            False, "",
        ),
        ("invalid governance key",
         dict(
             run_id="r1", dataset="own_full", method="V4", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
             governance=dict(rbac="on"),
         ), True, "FORBIDDEN config keys"),
        ("invalid LoCoMo dataset",
         dict(
             run_id="r1", dataset="locomo", method="V4", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
        ), True, "FORBIDDEN by the released protocol"),
        ("invalid max_queries subsample",
         dict(
             run_id="r1", dataset="own_full", method="V4", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
             max_queries=100,
         ), True, "FORBIDDEN config keys"),
        ("invalid low-tier silent fallback",
         dict(
             run_id="r1", dataset="own_full", method="V4", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_low",
                         answer="chat_low", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
         ), True, "silent substitute"),
        ("invalid path references LoCoMo",
         dict(
             run_id="r1", dataset="own_full", method="V4", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="/path/to/locomo10.json", queries="x", chroma="x",
                        output="x", manifest="x"),
             tokenizer="cl100k_base",
         ), True, "FORBIDDEN dataset reference"),
        ("invalid promotion on B1",
         dict(
             run_id="r1", dataset="own_full", method="B1", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
             promotion=dict(score_weights={}, decision_mode="llm", promotion_budget=100),
         ), True, "promotion config block is only valid"),
        ("valid B1 baseline",
         dict(
             run_id="r1", dataset="own_full", method="B1", seed=42,
             models=dict(hierarchy_low="chat_low", hierarchy_high="chat_high",
                         answer="chat_high", judge="judge"),
             paths=dict(corpus="x", queries="x", chroma="x", output="x", manifest="x"),
             tokenizer="cl100k_base",
         ), False, ""),
    ]

    failures = []
    for label, cfg, should_fail, want_sub in cases:
        try:
            validate_config(cfg)
            ok = not should_fail
            msg = "validated"
        except V4ConfigError as e:
            ok = should_fail and (want_sub.lower() in str(e).lower())
            msg = str(e).split("\n", 1)[0]
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {msg[:100]}")
        if not ok:
            failures.append((label, msg))
    if failures:
        print(f"\n{len(failures)} self-test failure(s):")
        for label, msg in failures:
            print(f"  - {label}: {msg}")
        return 1
    print(f"\nAll {len(cases)} self-tests PASSED.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
