"""Data contracts for document retrieval and evaluation.

Every derived record retains source chunk identifiers and section paths for
end-to-end evidence traceability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawChunkEntry:
    """Raw chunk preserved verbatim from a source document.

    chunk_id format: ``"{document_id}__sec_{section_idx}__chunk_{chunk_idx}"``
    """

    chunk_id: str
    document_id: str
    section_id: str
    section_path: str
    position: int
    raw_text: str
    domain: str
    source_type: str
    title: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Flatten for ChromaDB metadata (only str/int/float/bool allowed)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "position": self.position,
            "domain": self.domain,
            "source_type": self.source_type,
            "title": self.title,
            "token_count": self.token_count,
            "entry_kind": "raw_chunk",
        }


@dataclass
class DistilledMemoryEntry:
    """High-value knowledge extracted from raw chunks.

    Memory types (basic): fact / procedure / definition / requirement / decision.
    """

    memory_id: str  # deterministic from index hash
    memory_type: str
    index: str
    value: str
    document_id: str
    section_id: str
    section_path: str
    chunk_id: str  # primary source chunk
    source_chunk_ids: List[str] = field(default_factory=list)
    domain: str = ""
    source_type: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
            "source_chunk_ids": ",".join(self.source_chunk_ids) if self.source_chunk_ids else self.chunk_id,
            "domain": self.domain,
            "source_type": self.source_type,
            "confidence": float(self.confidence),
            "entry_kind": "distilled_memory",
        }


@dataclass
class CognitiveEntry:
    """Cognitive relation entry (CDM, 14 types).

    Supported relation types are:
    definition / fact / procedure / requirement / constraint / decision /
    dependency / causal / exception / risk / recommendation / example /
    conflict / version_update.
    """

    cognitive_id: str
    memory_type: str  # one of 14 cognitive types
    index: str  # subject + relation + object
    value: str
    subject: str = ""
    relation: str = ""
    object_: str = ""  # 'object' is a builtin
    condition: str = ""
    effect: str = ""
    version: str = ""
    document_id: str = ""
    section_id: str = ""
    section_path: str = ""
    chunk_id: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    domain: str = ""
    source_type: str = ""
    confidence: float = 1.0
    conflict_doc_ids: List[str] = field(default_factory=list)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "cognitive_id": self.cognitive_id,
            "memory_type": self.memory_type,
            "subject": self.subject[:200],
            "relation": self.relation[:100],
            "object_": self.object_[:200],
            "condition": self.condition[:300],
            "effect": self.effect[:300],
            "version": self.version[:100],
            "document_id": self.document_id,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
            "source_chunk_ids": ",".join(self.source_chunk_ids) if self.source_chunk_ids else self.chunk_id,
            "domain": self.domain,
            "source_type": self.source_type,
            "confidence": float(self.confidence),
            "conflict_doc_ids": ",".join(self.conflict_doc_ids),
            "entry_kind": "cognitive_entry",
        }


@dataclass
class SectionNode:
    section_id: str
    document_id: str
    section_path: str
    section_title: str
    level: int = 0
    summary: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    parent_section_id: str = ""
    prev_section_id: str = ""
    next_section_id: str = ""
    domain: str = ""
    source_type: str = ""
    token_count: int = 0

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "section_title": self.section_title[:200],
            "level": self.level,
            "chunk_ids": ",".join(self.chunk_ids[:50]),  # truncate for chroma
            "parent_section_id": self.parent_section_id,
            "prev_section_id": self.prev_section_id,
            "next_section_id": self.next_section_id,
            "domain": self.domain,
            "source_type": self.source_type,
            "token_count": self.token_count,
            "entry_kind": "section_summary",
        }


@dataclass
class DocumentNode:
    document_id: str
    title: str
    doc_type: str
    domain: str
    source_type: str
    section_ids: List[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title[:300],
            "doc_type": self.doc_type,
            "domain": self.domain,
            "source_type": self.source_type,
            "section_ids": ",".join(self.section_ids[:50]),
            "token_count": self.token_count,
            "entry_kind": "document_summary",
        }


@dataclass
class DocumentRetrievalConfig:
    """Master config for DocumentRetriever (Option A architecture).

    Initial evaluation runs can keep HDM routing disabled while retaining DDI
    and CDM. Larger stages enable ``document_routing_enabled`` explicitly.

    Toggles allow ablations:
    - DDI-only:    enable_cdm=False, enable_section_routing=False
    - HDM-only:    enable_cognitive_path=False, enable_distilled_stream=False
    - CDM-only:    enable_raw_stream=False, enable_distilled_stream=False, ...
    """

    # Algorithm enables (top-level)
    enable_dual_index: bool = True
    enable_hierarchical: bool = True
    enable_cdm: bool = True

    # DDI stream toggles
    enable_raw_stream: bool = True
    enable_distilled_stream: bool = True

    # HDM routing toggles
    document_routing_enabled: bool = False  # Stage 1 default OFF
    section_routing_enabled: bool = False  # Stage 1 default OFF
    section_expansion_depth: int = 1
    K_D: int = 10  # Document routing top-K
    K_S: int = 20  # Section routing top-K

    # CDM toggles
    enable_cognitive_path: bool = True
    relation_expansion_depth: int = 1
    primary_weight: float = 0.5
    expansion_weight: float = 0.3
    semantic_weight: float = 0.2

    # DDI retrieval params
    K_A: int = 20  # Raw chunk top-K
    K_B: int = 20  # Distilled memory top-K
    alpha: float = 0.5  # RRF weight: Raw vs Distilled
    frontier_window: int = 1

    # Final assembly
    top_n_final: int = 10
    llm_token_budget: int = 4096

    # Build params (used by pipeline)
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 50
    max_chunks_per_doc: int = 200
    distilled_memory_per_chunk_budget: int = 3
    cognitive_extraction_enabled: bool = True
    section_summary_min_tokens: int = 1000  # below this skip LLM, use raw text
    # Stage 2 Option E: combine distilled + cognitive into a single LLM call per chunk.
    # ~50% extract wall-time saving with similar token volume.
    use_combined_distilled_cognitive_prompt: bool = False

    # LLM config - general chat API.
    chat_model_id: str = "YOUR_CHAT_MODEL"
    judge_model_id: str = "YOUR_JUDGE_MODEL"
    llm_api_base: str = ""
    llm_api_key: str = ""
    max_completion_tokens: int = 800

    # Optional secondary general API endpoint for extraction-heavy runs. When
    # enabled, calls without an explicit model id use this route.
    use_secondary_api: bool = False
    secondary_api_base: str = "YOUR_SECONDARY_LLM_API_BASE"
    secondary_api_key: str = ""
    secondary_model_id: str = "YOUR_SECONDARY_LLM_MODEL"

    # Storage
    chroma_path: str = "./chroma_doc_eval"
    collection_prefix: str = "doc_eval"

    # Embeddings
    use_local_embedding: bool = True
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Eval / runtime
    seed: int = 42

    def collection_name(self, kind: str) -> str:
        """All Document collections include ``_doc_`` to physically isolate from LoCoMo."""
        return f"{self.collection_prefix}_doc_{kind}"
