"""
LLM-driven extractors for DDI distilled memories and CDM cognitive entries.

Both extractors return structured JSON; on parsing failure we silently fall
back to "no extractions" rather than crashing — this is consistent with
Solution2's AgentMemory distillation policy.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from megamem.document_eval.llm_clients import chat_completion
from megamem.document_eval.types import (
    CognitiveEntry,
    DistilledMemoryEntry,
    DocumentRetrievalConfig,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Distilled memory extraction (DDI Stream B)
# -------------------------------------------------------------------------

DISTILL_PROMPT_SYSTEM = """\
You extract atomic, retrieval-friendly distilled memories from an enterprise document chunk.

Return JSON with the schema:
{{
  "memories": [
    {{"memory_type": "fact" | "procedure" | "definition" | "requirement" | "decision",
     "index": "short 4-10 word retrieval key",
     "value": "1-3 sentence factual statement supported by the source"}}
  ]
}}

Rules:
- At most {budget} memories per chunk.
- "memory_type" must be one of: fact, procedure, definition, requirement, decision.
- "index" is a concise topic label that someone might query with.
- "value" must be entailed by the source chunk; do not add outside knowledge.
- Skip greetings, salutations, signatures, and trivial filler.
- If the chunk has nothing useful, return {{"memories": []}}.
"""


def extract_distilled_memories(
    cfg: DocumentRetrievalConfig,
    chunk_id: str,
    document_id: str,
    section_id: str,
    section_path: str,
    domain: str,
    source_type: str,
    raw_text: str,
) -> List[DistilledMemoryEntry]:
    """Call the chat model to extract distilled memories from a single chunk."""
    if not raw_text.strip():
        return []
    sys = DISTILL_PROMPT_SYSTEM.format(budget=cfg.distilled_memory_per_chunk_budget)
    try:
        resp = chat_completion(
            cfg,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Document chunk (section: {section_path}):\n\n{raw_text[:6000]}"},
            ],
            response_format_json=True,
        )
        data = json.loads(resp) if resp else {"memories": []}
    except Exception as exc:
        logger.warning(f"Distilled extraction failed for {chunk_id}: {exc}")
        return []

    memories = data.get("memories", []) if isinstance(data, dict) else []
    out: List[DistilledMemoryEntry] = []
    for i, m in enumerate(memories[: cfg.distilled_memory_per_chunk_budget]):
        if not isinstance(m, dict):
            continue
        mtype = str(m.get("memory_type", "fact")).strip().lower()
        if mtype not in {"fact", "procedure", "definition", "requirement", "decision"}:
            mtype = "fact"
        idx = str(m.get("index", "")).strip()
        val = str(m.get("value", "")).strip()
        if not idx or not val:
            continue
        out.append(
            DistilledMemoryEntry(
                memory_id=f"{chunk_id}__mem_{i}",
                memory_type=mtype,
                index=idx[:300],
                value=val[:2000],
                document_id=document_id,
                section_id=section_id,
                section_path=section_path,
                chunk_id=chunk_id,
                source_chunk_ids=[chunk_id],
                domain=domain,
                source_type=source_type,
                confidence=1.0,
            )
        )
    return out


# -------------------------------------------------------------------------
# Cognitive extraction (CDM, 14 relation types in a single LLM call)
# -------------------------------------------------------------------------

COGNITIVE_TYPES = [
    "definition",
    "fact",
    "procedure",
    "requirement",
    "constraint",
    "decision",
    "dependency",
    "causal",
    "exception",
    "risk",
    "recommendation",
    "example",
    "conflict",  # cross-doc; skip in single-chunk extraction
    "version_update",
]
COGNITIVE_SINGLE_CHUNK_TYPES = [t for t in COGNITIVE_TYPES if t != "conflict"]

COGNITIVE_PROMPT_SYSTEM = """\
You extract cognitive relations from an enterprise document chunk.

Return JSON:
{
  "relations": [
    {"memory_type": "<one of: definition, fact, procedure, requirement, constraint, decision, dependency, causal, exception, risk, recommendation, example, version_update>",
     "subject": "the entity or concept the relation is about",
     "relation": "short verb phrase, e.g. 'requires', 'causes', 'bypasses'",
     "object": "the entity or concept on the other side of the relation",
     "condition": "trigger condition if applicable, else empty",
     "effect": "result or impact if applicable, else empty",
     "version": "version number if memory_type=version_update, else empty",
     "value": "1-2 sentence full statement entailed by the source",
     "confidence": 0.0 to 1.0}
  ]
}

Rules:
- Extract at most 4 relations per chunk.
- Pick the SINGLE BEST memory_type for each relation; if unclear, use 'fact'.
- 'constraint': hard rule / policy / quota / limit.
- 'exception': bypasses a rule under a condition.
- 'dependency': A requires B to exist or function.
- 'causal': A causes / leads to B.
- 'risk': potential negative outcome with a trigger.
- 'requirement': must-have / shall-have.
- 'recommendation': should / best-practice (non-mandatory).
- 'decision': we chose to do X.
- 'version_update': a value/policy changed between versions.
- 'definition': formal term definition.
- 'procedure': ordered steps.
- 'example': concrete instance.
- 'fact': stand-alone neutral assertion that doesn't fit the categories above.
- Skip salutations, greetings, signatures, and pure filler.
- If nothing extractable, return {"relations": []}.
"""


def extract_cognitive_relations(
    cfg: DocumentRetrievalConfig,
    chunk_id: str,
    document_id: str,
    section_id: str,
    section_path: str,
    domain: str,
    source_type: str,
    raw_text: str,
) -> List[CognitiveEntry]:
    if not raw_text.strip():
        return []
    try:
        resp = chat_completion(
            cfg,
            messages=[
                {"role": "system", "content": COGNITIVE_PROMPT_SYSTEM},
                {"role": "user", "content": f"Document chunk (section: {section_path}):\n\n{raw_text[:6000]}"},
            ],
            response_format_json=True,
        )
        data = json.loads(resp) if resp else {"relations": []}
    except Exception as exc:
        logger.warning(f"Cognitive extraction failed for {chunk_id}: {exc}")
        return []

    rels = data.get("relations", []) if isinstance(data, dict) else []
    out: List[CognitiveEntry] = []
    for i, r in enumerate(rels[:4]):
        if not isinstance(r, dict):
            continue
        mt = str(r.get("memory_type", "fact")).strip().lower()
        if mt not in COGNITIVE_SINGLE_CHUNK_TYPES:
            mt = "fact"
        subj = str(r.get("subject", "")).strip()
        rel = str(r.get("relation", "")).strip()
        obj = str(r.get("object", "")).strip()
        val = str(r.get("value", "")).strip()
        try:
            conf = float(r.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        if not val:
            continue
        if conf < 0.5:  # confidence floor
            continue
        index = " | ".join(x for x in [subj, rel, obj] if x).strip() or val[:80]
        out.append(
            CognitiveEntry(
                cognitive_id=f"{chunk_id}__cog_{i}",
                memory_type=mt,
                index=index[:300],
                value=val[:2000],
                subject=subj,
                relation=rel,
                object_=obj,
                condition=str(r.get("condition", "")).strip(),
                effect=str(r.get("effect", "")).strip(),
                version=str(r.get("version", "")).strip(),
                document_id=document_id,
                section_id=section_id,
                section_path=section_path,
                chunk_id=chunk_id,
                source_chunk_ids=[chunk_id],
                domain=domain,
                source_type=source_type,
                confidence=conf,
            )
        )
    return out


# -------------------------------------------------------------------------
# Section / document summary
# -------------------------------------------------------------------------

SECTION_SUMMARY_PROMPT = """\
Write a 1-3 sentence summary of the following document section. Focus on the topic
and scope; do not include trivial details.

Section title: {title}
Section content:
{body}

Summary:"""


def summarize_section(
    cfg: DocumentRetrievalConfig,
    title: str,
    body: str,
    *,
    min_tokens_for_llm: int = 1000,
) -> str:
    """Generate or fall back for section summary.

    If section is short (< min_tokens_for_llm), use raw text (truncated) directly.
    """
    from megamem.document_eval.chunking import count_tokens

    if count_tokens(body) < min_tokens_for_llm:
        # Use raw text, truncate to ~150 tokens worth (chars approx)
        return (body[:600]).strip()
    try:
        return chat_completion(
            cfg,
            messages=[
                {
                    "role": "user",
                    "content": SECTION_SUMMARY_PROMPT.format(title=title or "(untitled)", body=body[:6000]),
                }
            ],
            max_tokens=200,
        ).strip()
    except Exception as exc:
        logger.warning(f"Section summary failed for '{title}': {exc}")
        return (body[:600]).strip()


DOCUMENT_SUMMARY_PROMPT = """\
Write a 2-4 sentence summary of the document below. Capture topic, document type, and main thrust.

Document title: {title}
Section summaries:
{sections}

Summary:"""


def summarize_document(
    cfg: DocumentRetrievalConfig,
    title: str,
    section_summaries: List[str],
) -> str:
    if not section_summaries:
        return title or ""
    # If only 1 section, use that summary directly.
    if len(section_summaries) == 1:
        return section_summaries[0]
    joined = "\n- " + "\n- ".join(s for s in section_summaries[:20])
    try:
        return chat_completion(
            cfg,
            messages=[{"role": "user", "content": DOCUMENT_SUMMARY_PROMPT.format(title=title or "(untitled)", sections=joined)}],
            max_tokens=250,
        ).strip()
    except Exception as exc:
        logger.warning(f"Document summary failed for '{title}': {exc}")
        return joined[:600]


# -------------------------------------------------------------------------
# Option E — combined distilled + cognitive extraction (Stage 2 optimization)
#
# Single LLM call per chunk producing BOTH distilled memories and cognitive
# relations. Halves the extract LLM calls (~50% wall-time saving) at the cost
# of larger prompt + larger output.
#
# Deployments can compare this combined path with the separate-call path before
# enabling it for a new collection.
# -------------------------------------------------------------------------

COMBINED_PROMPT_SYSTEM = """\
You extract TWO kinds of structured knowledge from an enterprise document chunk:

(A) DISTILLED MEMORIES — atomic, retrieval-friendly facts. memory_type one of:
    fact / procedure / definition / requirement / decision.

(B) COGNITIVE RELATIONS — typed subject-relation-object tuples. memory_type one of:
    definition / fact / procedure / requirement / constraint / decision /
    dependency / causal / exception / risk / recommendation / example /
    version_update.

Return JSON with schema:
{{
  "memories": [
    {{"memory_type": "<distilled type>",
      "index": "short 4-10 word retrieval key",
      "value": "1-3 sentence factual statement"}}
  ],
  "relations": [
    {{"memory_type": "<cognitive type>",
      "subject": "entity/concept",
      "relation": "short verb phrase",
      "object": "entity/concept",
      "condition": "trigger condition (or empty)",
      "effect": "result or impact (or empty)",
      "version": "version (only if memory_type=version_update; else empty)",
      "value": "1-2 sentence full statement",
      "confidence": 0.0 to 1.0}}
  ]
}}

Rules for BOTH lists:
- At most {distilled_budget} memories AND at most 4 relations per chunk.
- Skip salutations, signatures, filler.
- 'value' must be entailed by the source; do not add outside knowledge.
- For relations only: if unclear what cognitive type best fits, use 'fact'.
- Distilled memory_type values are a SUBSET of cognitive (fact/procedure/definition/requirement/decision).
  A relation can also be one of the other cognitive types (constraint/exception/dependency/causal/risk/recommendation/example/version_update).
- If the chunk has nothing useful, return {{"memories": [], "relations": []}}.

Quality bar (important):
- Distilled memories should be SELF-CONTAINED short snippets queryable by simple semantic search.
- Cognitive relations should be TYPED graph edges suitable for relation-aware retrieval.
- The same underlying fact may appear in both lists if it qualifies as both —
  that is intended; we use distilled for general retrieval and cognitive for
  typed reasoning.
"""


def extract_combined_distilled_and_cognitive(
    cfg: DocumentRetrievalConfig,
    chunk_id: str,
    document_id: str,
    section_id: str,
    section_path: str,
    domain: str,
    source_type: str,
    raw_text: str,
) -> tuple:
    """Single-LLM-call extractor that returns (distilled_list, cognitive_list).

    Mirrors the return shape of extract_distilled_memories + extract_cognitive_relations
    so callers can swap implementations behind a config flag.
    """
    if not raw_text.strip():
        return [], []
    sys_prompt = COMBINED_PROMPT_SYSTEM.format(
        distilled_budget=cfg.distilled_memory_per_chunk_budget
    )
    try:
        resp = chat_completion(
            cfg,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Document chunk (section: {section_path}):\n\n{raw_text[:6000]}"},
            ],
            response_format_json=True,
            max_tokens=1500,  # bumped vs single-task extractors to fit both outputs
        )
        data = json.loads(resp) if resp else {"memories": [], "relations": []}
    except Exception as exc:
        logger.warning(f"Combined extraction failed for {chunk_id}: {exc}")
        return [], []

    if not isinstance(data, dict):
        return [], []

    # --- Parse distilled list (same logic as extract_distilled_memories) ---
    distilled_out: List[DistilledMemoryEntry] = []
    for i, m in enumerate(data.get("memories", [])[: cfg.distilled_memory_per_chunk_budget]):
        if not isinstance(m, dict):
            continue
        mtype = str(m.get("memory_type", "fact")).strip().lower()
        if mtype not in {"fact", "procedure", "definition", "requirement", "decision"}:
            mtype = "fact"
        idx = str(m.get("index", "")).strip()
        val = str(m.get("value", "")).strip()
        if not idx or not val:
            continue
        distilled_out.append(
            DistilledMemoryEntry(
                memory_id=f"{chunk_id}__mem_{i}",
                memory_type=mtype,
                index=idx[:300],
                value=val[:2000],
                document_id=document_id,
                section_id=section_id,
                section_path=section_path,
                chunk_id=chunk_id,
                source_chunk_ids=[chunk_id],
                domain=domain,
                source_type=source_type,
                confidence=1.0,
            )
        )

    # --- Parse cognitive list (same logic as extract_cognitive_relations) ---
    cognitive_out: List[CognitiveEntry] = []
    for i, r in enumerate(data.get("relations", [])[:4]):
        if not isinstance(r, dict):
            continue
        mt = str(r.get("memory_type", "fact")).strip().lower()
        if mt not in COGNITIVE_SINGLE_CHUNK_TYPES:
            mt = "fact"
        subj = str(r.get("subject", "")).strip()
        rel = str(r.get("relation", "")).strip()
        obj = str(r.get("object", "")).strip()
        val = str(r.get("value", "")).strip()
        try:
            conf = float(r.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        if not val:
            continue
        if conf < 0.5:
            continue
        index = " | ".join(x for x in [subj, rel, obj] if x).strip() or val[:80]
        cognitive_out.append(
            CognitiveEntry(
                cognitive_id=f"{chunk_id}__cog_{i}",
                memory_type=mt,
                index=index[:300],
                value=val[:2000],
                subject=subj,
                relation=rel,
                object_=obj,
                condition=str(r.get("condition", "")).strip(),
                effect=str(r.get("effect", "")).strip(),
                version=str(r.get("version", "")).strip(),
                document_id=document_id,
                section_id=section_id,
                section_path=section_path,
                chunk_id=chunk_id,
                source_chunk_ids=[chunk_id],
                domain=domain,
                source_type=source_type,
                confidence=conf,
            )
        )

    return distilled_out, cognitive_out
