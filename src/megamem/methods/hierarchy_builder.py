"""MegaMem source-level dual-view node builder."""
from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import tiktoken

from .configs.model_resolver import resolve
from .dual_node import DualNode, NODE_STATE_LIGHT
from .token_ledger import (
    PHASE_DISTILLED_GEN,
    PHASE_HIERARCHY_BUILD,
    TokenLedger,
)

logger = logging.getLogger(__name__)


DISTILL_SYSTEM_PROMPT = """You are an enterprise memory summarizer. Given a long-form L0 memory record from a corporate dataset, produce a SHORT distilled summary that:

- captures the central topic, entities, time references, and any explicit decisions or values stated
- preserves enough downstream signal that a retrieval system can route relevant queries to the correct node
- is strictly SHORTER than the input (target: 15-40 words)
- contains NO speculation or invented content
- contains NO gold answer fields, ground truth, evidence_link kinds, or expected_doc_ids tokens

Return ONLY the summary sentence(s); no prefix, no JSON, no quotation marks."""

DISTILL_USER_TEMPLATE = """L0 record body:
---
{body}
---

Distilled summary:"""


# Module-level general client cache
_GENERAL_CLIENT = None
_ENC = None


def _get_enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _ensure_general_client():
    """Singleton general API client with an enlarged connection pool."""
    global _GENERAL_CLIENT
    if _GENERAL_CLIENT is not None:
        return _GENERAL_CLIENT
    from megamem.core.general_api import GeneralAPIClient, build_general_session
    spec = resolve("chat_low")
    api_key = _read_api_key(spec)
    base_url = spec.get("base_url") or os.environ.get(spec.get("base_url_env", "LLM_API_BASE"), "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    if not base_url:
        raise RuntimeError("LLM_API_BASE is not set")
    _GENERAL_CLIENT = GeneralAPIClient(
        base_url=base_url,
        api_key=api_key,
        session=build_general_session(pool_max=256, pool_connections=128),
        timeout=120.0,
        max_retries=3,
    )
    return _GENERAL_CLIENT


def _read_api_key(spec: Dict[str, Any]) -> str:
    """Read a key only from the environment named by the public alias spec."""
    return os.environ.get(spec.get("api_key_env", "LLM_API_KEY"), "")


def llm_distill_one(body: str, max_retries: int = 4) -> Dict[str, Any]:
    """Call the configured low-tier model for one distilled summary.

    Returns dict with:
        text, input_tokens, output_tokens, wall_seconds, success, error
    """
    spec = resolve("chat_low")
    client = _ensure_general_client()
    enc = _get_enc()
    user_prompt = DISTILL_USER_TEMPLATE.format(body=body[:6000])
    input_tokens_est = len(enc.encode(DISTILL_SYSTEM_PROMPT)) + len(enc.encode(user_prompt))

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=spec["model"],
                messages=[
                    {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
            )
            wall = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_t = getattr(usage, "prompt_tokens", input_tokens_est) if usage else input_tokens_est
            out_t = getattr(usage, "completion_tokens", len(enc.encode(text))) if usage else len(enc.encode(text))
            return {
                "text": text,
                "input_tokens": int(in_t),
                "output_tokens": int(out_t),
                "wall_seconds": wall,
                "success": True,
                "error": None,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            is_429 = "429" in msg or "ratelimit" in msg or "too many" in msg
            if attempt < max_retries and (is_429 or "timeout" in msg):
                backoff = (2 ** attempt) + random.random()
                logger.warning(
                    f"llm_distill_one retry {attempt+1}/{max_retries} after {type(exc).__name__}, sleeping {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            break
    return {
        "text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_seconds": 0.0,
        "success": False,
        "error": f"{type(last_err).__name__}: {str(last_err)[:200]}",
        "attempts": max_retries + 1,
    }


def build_l0_dualnodes(
    l0_records: List[Dict[str, Any]],
    *,
    ledger: TokenLedger,
    max_workers: int = 8,
    progress_every: int = 100,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    alias_status_tag: str = "",
) -> List[DualNode]:
    """Build a DualNode per L0 record. L0 records are dicts like
        {node_id, tenant_id, canonical_label, level_specific.raw_text,
         level_specific.evidence_span_id (or source_evidence_span_ids)}

    Returns a list of DualNodes (one per input record). Failed-to-distill
    records still produce a DualNode but with `distilled_text == ""` and the
    error captured in `extra["distill_error"]`. Contract validation will catch
    these nodes and fail the run.
    """
    enc = _get_enc()

    def _node_body_and_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
        label = rec.get("canonical_label", "") or ""
        ls = rec.get("level_specific", {}) or {}
        raw = ls.get("raw_text", "") if isinstance(ls, dict) else ""
        body = label + ("\n" + raw if raw else "")
        # Collect provenance: evidence_span_id from level_specific, else node_id self-ref
        ev_ids: List[str] = []
        if isinstance(ls, dict):
            esid = ls.get("evidence_span_id")
            if esid:
                ev_ids.append(str(esid))
        for sid in rec.get("source_evidence_span_ids", []) or []:
            ev_ids.append(str(sid))
        if not ev_ids:
            ev_ids = [rec.get("node_id", "")]
        return {"body": body, "evidence_ids": ev_ids}

    def _process_one(rec: Dict[str, Any]) -> DualNode:
        info = _node_body_and_meta(rec)
        body = info["body"]
        detailed_tokens = len(enc.encode(body)) if body else 0
        distilled = llm_distill_one(body) if body else {
            "text": "", "input_tokens": 0, "output_tokens": 0,
            "wall_seconds": 0.0, "success": False, "error": "empty body",
        }
        distilled_text = distilled["text"]
        distilled_tokens = len(enc.encode(distilled_text)) if distilled_text else 0
        ledger.record(
            phase=PHASE_DISTILLED_GEN, model_alias="chat_low",
            input_tokens=distilled["input_tokens"], output_tokens=distilled["output_tokens"],
            wall_seconds=distilled["wall_seconds"],
            node_id=rec.get("node_id", ""),
        )
        node = DualNode(
            node_id=rec.get("node_id", ""),
            level=rec.get("level", "L0"),
            tenant_id=rec.get("tenant_id", ""),
            distilled_text=distilled_text,
            detailed_text=body,
            distilled_tokens=distilled_tokens,
            detailed_tokens=detailed_tokens,
            source_evidence_ids=info["evidence_ids"],
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias="chat_low",
            distilled_text_model_status=alias_status_tag,
        )
        if not distilled["success"]:
            node.extra["distill_error"] = distilled["error"]
            node.extra["distill_attempts"] = distilled["attempts"]
        return node

    nodes: List[DualNode] = []
    n_total = len(l0_records)
    done = 0
    fails = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, rec) for rec in l0_records]
        for fut in as_completed(futures):
            n = fut.result()
            nodes.append(n)
            done += 1
            if "distill_error" in n.extra:
                fails += 1
            if done % progress_every == 0 or done == n_total:
                logger.info(
                    f"  hierarchy_build: {done}/{n_total} done ({fails} distill failures so far)"
                )
                if progress_cb:
                    progress_cb(done, n_total)
    logger.info(f"hierarchy_build complete: {done}/{n_total} nodes, {fails} distill failures")
    return nodes
