"""
DocumentRetriever — Option A architecture (HDM routing + DDI dual-stream + CDM).

Toggles allow ablations to cover Option B (Flat) and Option C (Cascade) behaviors:

  enable_hierarchical=False         → DDI + CDM only (HDM disabled)
  enable_dual_index=False           → CDM-only path
  enable_cdm=False                  → DDI + HDM only
  document_routing_enabled=False    → Stage 1 default; HDM routing OFF
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from megamem.document_eval.extractors import COGNITIVE_SINGLE_CHUNK_TYPES
from megamem.document_eval.llm_clients import chat_completion
from megamem.document_eval.storage import DocumentStorage
from megamem.document_eval.types import DocumentRetrievalConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Relation expansion adjacency for cognitive retrieval
# --------------------------------------------------------------------------

# Maps each source memory type to depth-one expansion targets.
RELATION_ADJACENCY: Dict[str, List[str]] = {
    "constraint": ["exception", "requirement", "decision", "dependency", "risk", "conflict", "version_update"],
    "exception": ["constraint", "procedure", "requirement", "example"],
    "dependency": ["procedure", "requirement", "constraint", "causal", "risk"],
    "causal": ["risk", "fact", "dependency"],
    "risk": ["causal", "requirement", "constraint", "decision", "dependency", "exception", "recommendation"],
    "version_update": ["fact", "procedure", "constraint", "decision"],
    "conflict": ["definition", "fact", "requirement", "constraint", "decision", "version_update"],
    "requirement": ["constraint", "exception", "decision"],
    "decision": ["requirement", "constraint"],
    "procedure": ["requirement", "constraint", "example"],
    "definition": ["fact", "example"],
    "fact": ["definition", "example"],
    "recommendation": ["fact", "example"],
    "example": ["fact", "definition"],
}


# --------------------------------------------------------------------------
# Query intent → primary cognitive types
# --------------------------------------------------------------------------

QUERY_INTENT_KEYWORDS: List[Tuple[List[str], List[str]]] = [
    (["can i", "am i allowed", "is it permitted", "allowed to", "permitted"],
     ["exception", "constraint"]),
    (["what happens if", "what causes", "why does", "why is"], ["causal", "risk"]),
    (["how to", "steps to", "procedure", "how do i"], ["procedure"]),
    (["must", "shall", "required", "requirement", "what is the requirement"],
     ["requirement", "constraint"]),
    (["what changed", "new version", "as of version", "in version"],
     ["version_update"]),
    (["conflict", "inconsistency", "contradict"], ["conflict"]),
    (["depends on", "requires", "prerequisite"], ["dependency"]),
    (["risk", "danger", "could go wrong"], ["risk", "causal"]),
]


def classify_query_intent(query: str) -> Tuple[List[str], List[str]]:
    """Return (primary_types, secondary_types) for a query.

    This classifier is deterministic and makes no model calls.
    Fallback: ('fact', 'definition') + ('recommendation', 'example') for generic.
    """
    q = query.lower()
    for kws, primary in QUERY_INTENT_KEYWORDS:
        if any(k in q for k in kws):
            # Secondary: union of adjacency targets of each primary, deduped, capped.
            secondary: List[str] = []
            for p in primary:
                for t in RELATION_ADJACENCY.get(p, []):
                    if t not in primary and t not in secondary:
                        secondary.append(t)
            return primary, secondary[:5]
    return ["fact", "definition"], ["recommendation", "example"]


# --------------------------------------------------------------------------
# RRF helper
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float, Dict[str, Any]]]],
    weights: Optional[List[float]] = None,
    k_const: int = 60,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Standard RRF over multiple ranked lists.

    Each list is [(id, score, payload), ...]; score is ignored, rank is used.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: Dict[str, float] = defaultdict(float)
    payload_by_id: Dict[str, Dict[str, Any]] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, (_id, _score, payload) in enumerate(lst):
            fused[_id] += w / (k_const + rank + 1)
            if _id not in payload_by_id:
                payload_by_id[_id] = payload
    sorted_ids = sorted(fused.keys(), key=lambda i: -fused[i])
    return [(i, fused[i], payload_by_id[i]) for i in sorted_ids]


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Structured retrieval output: ranked chunks + metadata."""

    query: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # final ranked chunk dicts
    documents_retrieved: List[str] = field(default_factory=list)
    primary_cognitive_types: List[str] = field(default_factory=list)
    secondary_cognitive_types: List[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    trace: Dict[str, Any] = field(default_factory=dict)


class DocumentRetriever:
    """Option A retriever with toggles for ablations."""

    def __init__(self, cfg: DocumentRetrievalConfig, storage: Optional[DocumentStorage] = None):
        self.cfg = cfg
        self.storage = storage or DocumentStorage(cfg)

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.time()
        result = RetrievalResult(query=query)
        trace: Dict[str, Any] = {}
        cfg = self.cfg

        # --- Step 1: Cognitive intent analysis (cheap, always done if CDM enabled) ---
        primary_types: List[str] = []
        secondary_types: List[str] = []
        if cfg.enable_cdm and cfg.enable_cognitive_path:
            primary_types, secondary_types = classify_query_intent(query)
            result.primary_cognitive_types = primary_types
            result.secondary_cognitive_types = secondary_types

        # --- Step 2: HDM routing (Stage 1 default OFF; Flat fallback) ---
        candidate_doc_ids: Optional[List[str]] = None
        candidate_section_ids: Optional[List[str]] = None
        if cfg.enable_hierarchical and cfg.document_routing_enabled:
            try:
                candidate_doc_ids = self._route_documents(query, top_k=cfg.K_D)
                trace["doc_routing_top_k"] = candidate_doc_ids[:5]
            except Exception as exc:
                logger.warning(f"Document routing failed: {exc}")

        if cfg.enable_hierarchical and cfg.section_routing_enabled:
            try:
                where = {"document_id": {"$in": candidate_doc_ids}} if candidate_doc_ids else None
                candidate_section_ids = self._route_sections(query, top_k=cfg.K_S, where=where)
                trace["section_routing_top_k"] = candidate_section_ids[:5]
            except Exception as exc:
                logger.warning(f"Section routing failed: {exc}")

        # --- Step 3: DDI dual-stream retrieval ---
        ranked_raw: List[Tuple[str, float, Dict[str, Any]]] = []
        ranked_distilled: List[Tuple[str, float, Dict[str, Any]]] = []
        if cfg.enable_dual_index and cfg.enable_raw_stream:
            ranked_raw = self._query_collection(
                "raw_chunks",
                query=query,
                n_results=cfg.K_A,
                where=self._build_where(candidate_doc_ids, candidate_section_ids),
            )
            trace["raw_stream_n"] = len(ranked_raw)
        if cfg.enable_dual_index and cfg.enable_distilled_stream:
            ranked_distilled = self._query_collection(
                "distilled_memory",
                query=query,
                n_results=cfg.K_B,
                where=self._build_where(candidate_doc_ids, candidate_section_ids),
            )
            trace["distilled_stream_n"] = len(ranked_distilled)

        # --- Step 4: CDM cognitive retrieval ---
        ranked_cognitive_primary: List[Tuple[str, float, Dict[str, Any]]] = []
        ranked_cognitive_expansion: List[Tuple[str, float, Dict[str, Any]]] = []
        if cfg.enable_cdm and cfg.enable_cognitive_path and primary_types:
            try:
                ranked_cognitive_primary = self._query_collection(
                    "cognitive",
                    query=query,
                    n_results=15,
                    where=self._combine_where(
                        self._build_where(candidate_doc_ids, candidate_section_ids),
                        {"memory_type": {"$in": primary_types}},
                    ),
                )
                trace["cognitive_primary_n"] = len(ranked_cognitive_primary)
                if cfg.relation_expansion_depth >= 1 and secondary_types:
                    ranked_cognitive_expansion = self._query_collection(
                        "cognitive",
                        query=query,
                        n_results=10,
                        where=self._combine_where(
                            self._build_where(candidate_doc_ids, candidate_section_ids),
                            {"memory_type": {"$in": secondary_types}},
                        ),
                    )
                    trace["cognitive_expansion_n"] = len(ranked_cognitive_expansion)
            except Exception as exc:
                logger.warning(f"Cognitive retrieval failed: {exc}")

        # --- Step 5: Combine all into final chunk list ---
        # All candidates need to map back to raw chunks (the final evidence unit).
        chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_meta: Dict[str, Dict[str, Any]] = {}
        ranked_lists: List[List[Tuple[str, float, Dict[str, Any]]]] = []
        weights: List[float] = []

        def _to_chunk_list(
            entries: List[Tuple[str, float, Dict[str, Any]]]
        ) -> List[Tuple[str, float, Dict[str, Any]]]:
            """Convert retrieved entries (any kind) into a ranked list keyed by chunk_id."""
            out: List[Tuple[str, float, Dict[str, Any]]] = []
            seen: set = set()
            for _id, score, payload in entries:
                meta = payload.get("metadata", {}) or {}
                # Resolve to chunk_id (raw_chunks index by chunk_id; others have chunk_id in meta)
                chunk_id = meta.get("chunk_id") or _id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                # Also propagate source_chunk_ids (comma-sep) when present
                extra_chunks = meta.get("source_chunk_ids", "") or ""
                if extra_chunks and isinstance(extra_chunks, str):
                    for cid in extra_chunks.split(","):
                        cid = cid.strip()
                        if cid and cid not in seen:
                            seen.add(cid)
                            out.append((cid, score, {"metadata": meta}))
                out.append((chunk_id, score, payload))
            return out

        if cfg.enable_dual_index:
            if cfg.enable_raw_stream and ranked_raw:
                ranked_lists.append(_to_chunk_list(ranked_raw))
                weights.append(cfg.alpha)
            if cfg.enable_distilled_stream and ranked_distilled:
                ranked_lists.append(_to_chunk_list(ranked_distilled))
                weights.append(1.0 - cfg.alpha)
        if cfg.enable_cdm and cfg.enable_cognitive_path:
            if ranked_cognitive_primary:
                ranked_lists.append(_to_chunk_list(ranked_cognitive_primary))
                weights.append(cfg.primary_weight)
            if ranked_cognitive_expansion:
                ranked_lists.append(_to_chunk_list(ranked_cognitive_expansion))
                weights.append(cfg.expansion_weight)

        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)

        # Take top-N chunk_ids and fetch raw text from raw_chunks collection
        top_ids = [cid for cid, _, _ in fused[: cfg.top_n_final * 2]]  # over-fetch then trim
        chunk_docs_map = self._fetch_chunks(top_ids)

        final_chunks: List[Dict[str, Any]] = []
        for cid, score, payload in fused:
            if cid not in chunk_docs_map:
                continue
            cd = chunk_docs_map[cid]
            final_chunks.append({
                "chunk_id": cid,
                "score": float(score),
                "document_id": cd.get("document_id", ""),
                "section_id": cd.get("section_id", ""),
                "section_path": cd.get("section_path", ""),
                "raw_text": cd.get("raw_text", ""),
                "source_type": cd.get("source_type", ""),
            })
            if len(final_chunks) >= cfg.top_n_final:
                break

        result.chunks = final_chunks
        result.documents_retrieved = list({c["document_id"] for c in final_chunks if c["document_id"]})
        result.trace = trace
        result.retrieval_seconds = time.time() - t0
        return result

    # ----------------------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------------------

    def _route_documents(self, query: str, top_k: int) -> List[str]:
        ranked = self._query_collection("doc_summaries", query=query, n_results=top_k)
        return [_id for _id, _, _ in ranked]

    def _route_sections(
        self, query: str, top_k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        ranked = self._query_collection("section_summaries", query=query, n_results=top_k, where=where)
        return [_id for _id, _, _ in ranked]

    def _query_collection(
        self,
        kind: str,
        *,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query a chroma collection and return [(id, sim_score, payload)] in rank order."""
        if self.storage.count(kind) == 0:
            return []
        res = self.storage.query(
            kind,
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        ids_list = (res.get("ids") or [[]])[0]
        meta_list = (res.get("metadatas") or [[]])[0]
        docs_list = (res.get("documents") or [[]])[0]
        dists_list = (res.get("distances") or [[]])[0]

        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for _id, _meta, _doc, _dist in zip(ids_list, meta_list, docs_list, dists_list):
            sim = 1.0 - float(_dist) if _dist is not None else 0.0
            out.append((_id, sim, {"metadata": _meta or {}, "document": _doc}))
        return out

    def _build_where(
        self, doc_ids: Optional[List[str]], section_ids: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Build a Chroma 'where' filter for HDM-routed candidates."""
        clauses = []
        if section_ids:
            clauses.append({"section_id": {"$in": section_ids}})
        elif doc_ids:
            clauses.append({"document_id": {"$in": doc_ids}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _combine_where(
        a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not a and not b:
            return None
        if a and not b:
            return a
        if b and not a:
            return b
        return {"$and": [a, b]}

    def _fetch_chunks(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch chunk raw text + metadata by chunk_id."""
        out: Dict[str, Dict[str, Any]] = {}
        if not chunk_ids:
            return out
        try:
            got = self.storage.get_by_ids("raw_chunks", chunk_ids)
        except Exception as exc:
            logger.warning(f"Failed to fetch chunks: {exc}")
            return out
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        docs = got.get("documents") or []
        for cid, meta, doc in zip(ids, metas, docs):
            m = meta or {}
            out[cid] = {
                "chunk_id": cid,
                "document_id": m.get("document_id", ""),
                "section_id": m.get("section_id", ""),
                "section_path": m.get("section_path", ""),
                "source_type": m.get("source_type", ""),
                "raw_text": doc or "",
            }
        return out
