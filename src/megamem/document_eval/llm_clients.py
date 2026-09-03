"""General model and embedding clients for document_eval.

The artifact exposes one public provider type: a general JSON model gateway
configured through environment variables or DocumentRetrievalConfig fields.
An optional secondary general API endpoint can be enabled for extraction-heavy
runs, but it follows the same client contract as the primary route.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from megamem.core.general_api import GeneralAPIClient, build_general_session
from megamem.document_eval.types import DocumentRetrievalConfig

logger = logging.getLogger(__name__)


_LOCAL_ST_MODEL: Dict[str, Any] = {}
_GENERAL_CLIENTS: Dict[Tuple[str, str], GeneralAPIClient] = {}


def _make_http_client(prefix: str = "LLM"):
    """Create a requests session with tunable pools for parallel builds."""
    timeout_seconds = float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "120"))
    pool_max = int(os.getenv(f"{prefix}_POOL_MAX", "256"))
    pool_keepalive = int(os.getenv(f"{prefix}_POOL_KEEPALIVE", "128"))
    return (
        build_general_session(pool_max=pool_max, pool_connections=pool_keepalive),
        timeout_seconds,
    )


def get_general_client(
    *,
    api_base: str,
    api_key: str,
    env_base: str,
    env_key: str,
    pool_prefix: str,
) -> GeneralAPIClient:
    """Build or fetch a cached general chat API client."""
    base_url = api_base or os.getenv(env_base, "")
    key = api_key or os.getenv(env_key, "")
    if not base_url or not key:
        raise RuntimeError(
            "General LLM API base/key not configured. "
            f"Set {env_base} and {env_key}, or pass them through the run config."
        )

    cache_key = (base_url, key)
    cached = _GENERAL_CLIENTS.get(cache_key)
    if cached is not None:
        return cached

    session, timeout = _make_http_client(pool_prefix)
    client = GeneralAPIClient(
        base_url=base_url,
        api_key=key,
        session=session,
        timeout=timeout,
        max_retries=2,
    )
    _GENERAL_CLIENTS[cache_key] = client
    return client


def get_primary_client(cfg: DocumentRetrievalConfig) -> GeneralAPIClient:
    """Return the primary general API client."""
    return get_general_client(
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        env_base="LLM_API_BASE",
        env_key="LLM_API_KEY",
        pool_prefix="LLM",
    )


def get_secondary_client(cfg: DocumentRetrievalConfig) -> GeneralAPIClient:
    """Return the optional secondary general API client."""
    return get_general_client(
        api_base=cfg.secondary_api_base,
        api_key=cfg.secondary_api_key,
        env_base="SECONDARY_LLM_API_BASE",
        env_key="SECONDARY_LLM_API_KEY",
        pool_prefix="SECONDARY_LLM",
    )


def chat_completion(
    cfg: DocumentRetrievalConfig,
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format_json: bool = False,
) -> str:
    """Run one chat completion through the configured general API.

    If ``cfg.use_secondary_api`` is true and no explicit model is passed,
    extraction/generation calls use the secondary route. Explicit model calls,
    such as judge calls, always use the primary route.
    """
    if model is None and cfg.use_secondary_api:
        client = get_secondary_client(cfg)
        model = cfg.secondary_model_id or os.getenv("SECONDARY_LLM_MODEL", "")
    else:
        client = get_primary_client(cfg)
        model = model or cfg.chat_model_id or os.getenv("LLM_CHAT_MODEL", "")

    if not model or model.startswith("YOUR_"):
        raise RuntimeError("A concrete LLM model id must be configured before running.")

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_completion_tokens,
        "temperature": temperature,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def get_local_embedder(cfg: DocumentRetrievalConfig):
    """Return cached sentence-transformers model."""
    name = cfg.local_embedding_model
    m = _LOCAL_ST_MODEL.get(name)
    if m is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedder: {name}")
        m = SentenceTransformer(name)
        _LOCAL_ST_MODEL[name] = m
    return m


def embed_texts(
    cfg: DocumentRetrievalConfig,
    texts: List[str],
    batch_size: int = 64,
) -> List[List[float]]:
    """Embed a list of strings."""
    if not texts:
        return []
    model = get_local_embedder(cfg)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vecs]
