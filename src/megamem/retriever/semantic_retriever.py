"""
Pure-similarity retrieval strategy.

Thin wrapper around ``AgentMemory.query`` that performs a single semantic
(or hybrid) lookup with no planning, reformulation, or iteration.
"""

import time
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from megamem.retriever.base_retriever import BaseMemoryRetriever
from megamem.core.memory_entry import MemoryEntry
from megamem.core.memory import AgentMemory, QueryMode


class SemanticRetriever(BaseMemoryRetriever):
    """Vector-similarity backed memory retrieval."""

    def __init__(
        self,
        cfg: DictConfig,
        memory_client: Optional[AgentMemory] = None,
    ):
        """
        Build the semantic retriever from the shared configuration.

        Args:
            cfg: Configuration object
            memory_client: Optional pre-initialized memory client
            user_id: User identifier
        """
        super().__init__(cfg)
        self.memory_client = memory_client

        self.top_k = self.cfg.memory.get("top_k", 30)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = self.cfg.retrieval.get("enable_llm_filter", False)

        if self.cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_hybrid_search: Optional[bool] = None,
        enable_llm_filter: Optional[bool] = None,
        query_mode: Optional[QueryMode] = None,
        latency_tracker = None,
        **kwargs
    ) -> List[MemoryEntry]:
        """Issue a single semantic query against the memory store."""
        # Fall back to configured defaults when callers leave overrides empty.
        if top_k is None:
            top_k = self.top_k
        if enable_hybrid_search is None:
            enable_hybrid_search = self.enable_hybrid_search
        if enable_llm_filter is None:
            enable_llm_filter = self.enable_llm_filter
        if query_mode is None:
            query_mode = self.query_mode

        # Restrict to factual entries; episodic memories are reached
        # transitively through their links rather than searched directly.
        return self.memory_client.query(
            query,
            top_k=top_k,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            query_mode=query_mode,
            where={"memory_type": {"$eq": "factual"}},
            latency_tracker=latency_tracker,
        )
