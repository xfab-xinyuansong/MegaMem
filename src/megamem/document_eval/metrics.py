"""5 metrics for EnterpriseRAG-style document eval:"""
from __future__ import annotations

import json
import logging
import re
import string
from typing import Any, Dict, List, Optional

from megamem.document_eval.llm_clients import chat_completion
from megamem.document_eval.types import DocumentRetrievalConfig

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    return _normalize(s).split()


def bleu_score(pred: str, gold: str) -> float:
    """Smoothed sentence-BLEU using nltk; matches LoCoMo's metrics/utils.py."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return 0.0
    if not pred or not gold:
        return 0.0
    ref = [_tokens(gold)]
    hyp = _tokens(pred)
    if not hyp:
        return 0.0
    try:
        return float(sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1))
    except Exception:
        return 0.0


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1, common in QA evaluation."""
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    matched = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            matched += 1
            common[t] -= 1
    if matched == 0:
        return 0.0
    precision = matched / len(pred_toks)
    recall = matched / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


JUDGE_PROMPT = """\
You are a strict evaluator. Decide whether the model's RESPONSE answers the QUESTION correctly
according to the GOLD ANSWER.

Rules:
- Answer "yes" if RESPONSE conveys the same facts / decision / value as GOLD ANSWER,
  even if wording differs.
- Answer "yes" if RESPONSE is a superset of GOLD ANSWER's facts.
- Answer "no" if RESPONSE contradicts GOLD ANSWER, omits a critical detail, or is "I don't have enough information".
- Off-by-one in counts/days is acceptable.

Return JSON:
{{"score": 0 or 1, "reasoning": "one short sentence"}}

QUESTION: {question}

GOLD ANSWER: {gold}

RESPONSE: {pred}

JSON:"""


def llm_judge_score(
    cfg: DocumentRetrievalConfig,
    question: str,
    gold: str,
    pred: str,
) -> Dict[str, Any]:
    """LLM-as-judge scoring with the configured judge model."""
    if not pred or pred.strip().lower() == "i don't have enough information to answer.":
        return {"score": 0, "reasoning": "no answer"}
    try:
        resp = chat_completion(
            cfg,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
            model=cfg.judge_model_id,
            response_format_json=True,
            max_tokens=200,
        )
        data = json.loads(resp) if resp else {"score": 0, "reasoning": "parse_fail"}
        score = int(data.get("score", 0))
        return {"score": 1 if score >= 1 else 0, "reasoning": str(data.get("reasoning", ""))[:200]}
    except Exception as exc:
        logger.warning(f"Judge call failed: {exc}")
        return {"score": 0, "reasoning": f"judge_error:{type(exc).__name__}"}


def doc_recall(retrieved_doc_ids: List[str], expected_doc_ids: List[str]) -> float:
    """|retrieved ∩ expected| / |expected|. NaN-safe.

    "doc_recall replaces session_recall for EnterpriseRAG
    which has no session concept".
    """
    expected = set(expected_doc_ids or [])
    retrieved = set(retrieved_doc_ids or [])
    if not expected:
        return 0.0
    return len(retrieved & expected) / len(expected)


def _facts_present(fact: str, evidence_text: str, min_overlap: float = 0.5) -> bool:
    """Heuristic: fact's content tokens overlap with evidence by >= min_overlap fraction."""
    f_tokens = set(_tokens(fact))
    e_tokens = set(_tokens(evidence_text))
    if not f_tokens:
        return False
    overlap = len(f_tokens & e_tokens) / len(f_tokens)
    return overlap >= min_overlap


def text_recall(
    answer_facts: List[str],
    evidence_text: str,
    *,
    fallback_gold: str = "",
) -> float:
    """Fraction of answer_facts covered by evidence_text (token-overlap heuristic).

    If answer_facts is empty, fall back to gold-vs-evidence token recall:
        |gold_tokens ∩ evidence_tokens| / |gold_tokens|.
    """
    if answer_facts:
        if not evidence_text:
            return 0.0
        present = sum(1 for f in answer_facts if _facts_present(f, evidence_text))
        return present / len(answer_facts)
    # Fallback: gold-token recall against evidence
    g_tokens = set(_tokens(fallback_gold))
    e_tokens = set(_tokens(evidence_text))
    if not g_tokens:
        return 0.0
    return len(g_tokens & e_tokens) / len(g_tokens)
