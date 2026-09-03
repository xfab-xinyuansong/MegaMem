"""
Answer generation from retrieved chunks, with fixed token budget.

Constructs a prompt of [top-N chunks] -> question -> answer. Token budget is
enforced by truncating the chunk list (in rank order) until budget fits.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import tiktoken

from megamem.document_eval.llm_clients import chat_completion
from megamem.document_eval.types import DocumentRetrievalConfig

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


ANSWER_PROMPT_SYSTEM = """\
You are an enterprise knowledge assistant. Answer the user's question using ONLY \
the provided document evidence chunks. Be concise (1-4 sentences). If the evidence \
does not contain enough information to answer, reply exactly: "I don't have enough information to answer."."""


ANSWER_PROMPT_USER_TEMPLATE = """\
Question: {question}

Evidence (each labeled with chunk_id, document_id, section_path):
{chunks}

Answer:"""


def build_evidence_block(chunks: List[Dict[str, Any]], token_budget: int) -> Tuple[str, List[str]]:
    """Pack chunks into prompt up to token_budget. Return (text, used_chunk_ids)."""
    parts: List[str] = []
    used_ids: List[str] = []
    used_tokens = 0
    for c in chunks:
        text = c.get("raw_text", "")
        if not text:
            continue
        label = f"[chunk_id={c.get('chunk_id','')} | doc={c.get('document_id','')} | sec={c.get('section_path','')[:80]}]"
        block = f"{label}\n{text}\n"
        block_tokens = len(_ENC.encode(block))
        if used_tokens + block_tokens > token_budget:
            # Try to fit a truncated version of the chunk
            remaining = token_budget - used_tokens
            if remaining < 100:
                break
            truncated_tokens = _ENC.encode(text)[: max(remaining - 40, 0)]
            truncated_text = _ENC.decode(truncated_tokens)
            block = f"{label}\n{truncated_text}\n...[truncated]"
            parts.append(block)
            used_ids.append(c.get("chunk_id", ""))
            used_tokens = token_budget
            break
        parts.append(block)
        used_ids.append(c.get("chunk_id", ""))
        used_tokens += block_tokens
    return "\n".join(parts), used_ids


def generate_answer(
    cfg: DocumentRetrievalConfig,
    question: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a model answer + return diagnostic info."""
    evidence, used_ids = build_evidence_block(chunks, token_budget=cfg.llm_token_budget)
    user_prompt = ANSWER_PROMPT_USER_TEMPLATE.format(question=question, chunks=evidence or "(no evidence)")
    if not chunks:
        return {
            "answer": "I don't have enough information to answer.",
            "used_chunk_ids": [],
            "prompt_tokens": len(_ENC.encode(user_prompt)),
            "had_evidence": False,
        }
    try:
        answer = chat_completion(
            cfg,
            messages=[
                {"role": "system", "content": ANSWER_PROMPT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ).strip()
    except Exception as exc:
        logger.warning(f"Answer generation failed: {exc}")
        answer = "I don't have enough information to answer."
    return {
        "answer": answer,
        "used_chunk_ids": used_ids,
        "prompt_tokens": len(_ENC.encode(user_prompt)),
        "had_evidence": bool(chunks),
    }
