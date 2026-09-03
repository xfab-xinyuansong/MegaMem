"""Model alias resolution for MegaMem.

Configuration is loaded from ``MEGAMEM_MODELS_CONFIG`` when set, then from
``configs/models.yaml`` in the current project, and finally from the packaged
template. Missing aliases, unresolved providers, and placeholder model names
fail loudly so runs cannot silently drift across model families.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ModelAliasError(RuntimeError):
    pass


CONFIG_ENV_VAR = "MEGAMEM_MODELS_CONFIG"
CONFIG_DIR = Path(__file__).parent
PACKAGED_MODELS_YAML = CONFIG_DIR / "models.yaml"

_CACHE: Dict[Path, Dict[str, Any]] = {}


def _load_project_env() -> None:
    """Load a local ``.env`` when the optional LLM dependencies are installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def models_config_path() -> Path:
    """Return the model-alias configuration selected for this process."""
    _load_project_env()
    configured = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise ModelAliasError(
                f"{CONFIG_ENV_VAR} points to a missing file: {path}"
            )
        return path

    project_config = Path.cwd() / "configs" / "models.yaml"
    if project_config.is_file():
        return project_config.resolve()

    return PACKAGED_MODELS_YAML


def clear_config_cache() -> None:
    """Clear parsed model configs, primarily after changing config at runtime."""
    _CACHE.clear()


def _load_models_yaml() -> Dict[str, Any]:
    path = models_config_path()
    if path not in _CACHE:
        with path.open(encoding="utf-8") as handle:
            _CACHE[path] = yaml.safe_load(handle) or {}
    return _CACHE[path]


def resolve(alias: str) -> Dict[str, Any]:
    """Return the spec for an alias."""
    cfg = _load_models_yaml()
    if not isinstance(cfg, dict):
        raise ModelAliasError(
            f"model config must contain a YAML mapping: {models_config_path()}"
        )
    aliases = cfg.get("aliases", {})
    if not isinstance(aliases, dict):
        raise ModelAliasError(
            f"'aliases' must be a YAML mapping: {models_config_path()}"
        )
    if alias not in aliases:
        raise ModelAliasError(
            f"unknown model alias {alias!r}. Known: {sorted(aliases.keys())}"
        )
    spec = aliases[alias]
    if not isinstance(spec, dict):
        raise ModelAliasError(
            f"alias {alias!r} must contain a YAML mapping in {models_config_path()}"
        )
    if str(spec.get("provider", "")).upper() == "UNRESOLVED":
        raise ModelAliasError(
            f"alias {alias!r} is UNRESOLVED. "
            f"Blocker: {spec.get('blocker', 'unspecified')}."
        )
    if str(spec.get("provider", "")).lower() != "general":
        raise ModelAliasError(
            f"alias {alias!r} uses provider={spec.get('provider')!r}; "
            "only provider='general' is accepted by this artifact."
        )
    model = str(spec.get("model", "")).strip()
    if not model or model.upper().startswith("YOUR_"):
        raise ModelAliasError(
            f"alias {alias!r} has no concrete model in {models_config_path()}. "
            f"Copy configs/models.example.yaml to configs/models.yaml and replace "
            f"the placeholders, or set {CONFIG_ENV_VAR} to a completed config."
        )
    return spec


def assert_active(alias: str) -> None:
    """Raise if an alias is unavailable."""
    resolve(alias)


def _env_value(spec: Dict[str, Any], key: str, default: str = "") -> str:
    env_name = spec.get(key, "")
    return os.environ.get(env_name, default) if env_name else default


def smoke_test_general(
    alias: str,
    spec: Dict[str, Any],
    message: str = "Say 'pong' and nothing else.",
) -> Dict[str, Any]:
    """Perform a one-message chat API smoke test for an alias."""
    from megamem.core.general_api import GeneralAPIClient

    base_url = spec.get("base_url") or _env_value(spec, "base_url_env")
    api_key = spec.get("api_key") or _env_value(spec, "api_key_env")
    model = spec.get("model", "")
    if not base_url or not api_key:
        raise ModelAliasError(
            f"no API base/key for {alias} "
            f"(base_env={spec.get('base_url_env')}, key_env={spec.get('api_key_env')})"
        )
    if not model or model.startswith("YOUR_"):
        raise ModelAliasError(f"no concrete model configured for alias {alias}")

    client = GeneralAPIClient(base_url=base_url, api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.0,
    }
    token_param = spec.get("token_parameter", "max_tokens")
    kwargs[token_param] = 20

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""
    return {
        "alias": alias,
        "provider": "general",
        "model": model,
        "base_url": base_url,
        "reply": text,
        "latency_seconds": round(latency, 3),
        "ok": True,
    }


def smoke_test(alias: str) -> Dict[str, Any]:
    spec = resolve(alias)
    return smoke_test_general(alias, spec)


def list_aliases() -> Dict[str, str]:
    """Return {alias: status} for diagnostics."""
    cfg = _load_models_yaml()
    if not isinstance(cfg, dict) or not isinstance(cfg.get("aliases", {}), dict):
        raise ModelAliasError(
            f"model config must contain an 'aliases' mapping: {models_config_path()}"
        )
    out = {}
    for alias, spec in cfg.get("aliases", {}).items():
        if not isinstance(spec, dict):
            out[alias] = "INVALID (expected a mapping)"
            continue
        if str(spec.get("provider", "")).upper() == "UNRESOLVED":
            out[alias] = f"UNRESOLVED ({spec.get('blocker', '?')})"
        elif not str(spec.get("model", "")).strip() or str(spec.get("model", "")).upper().startswith("YOUR_"):
            out[alias] = f"PLACEHOLDER ({spec.get('model', 'missing model')})"
        else:
            out[alias] = f"{spec.get('provider')}::{spec.get('model')}"
    return out
