"""
Abstract retrieval surface for memory stores.

Defines the common interface that every retrieval strategy (semantic, hybrid,
plan-based, RL-driven, etc.) must satisfy so the rest of the stack can stay
agnostic to which approach is in use.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from megamem.core.memory_entry import MemoryEntry


class BaseMemoryRetriever(ABC):
    """
    Common base class for all memory-retrieval strategies.

    Concrete subclasses span a wide range of approaches — from a single
    vector-similarity lookup to multi-step reinforcement-learning pipelines —
    but they all expose the same minimal contract.

    What a subclass is expected to do:
    - Run retrieval against the underlying memory backend.
    - Apply whatever strategy the subclass implements (semantic, hybrid,
      RL, etc.).
    - Filter / rank the candidate memories before returning them.
    - Read its own knobs and hyper-parameters from the shared config.

    Example:
        class MyRetriever(BaseMemoryRetrieval):
            def retrieve(self, query: str, **kwargs) -> RetrievalResult:
                # custom logic goes here
                memories = self._fetch_memories(query)
                return RetrievalResult(
                    memories=memories,
                    query=query,
                    retrieval_time=elapsed_time,
                    strategy="my_strategy"
                )
    """

    def __init__(self, cfg: DictConfig):
        """
        Construct the retriever and remember the shared configuration.

        Args:
            cfg: Configuration object containing retrieval settings
            user_id: Optional user identifier for user-specific retrieval
        """
        self.cfg = cfg

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[MemoryEntry]:
        """
        Run a retrieval pass for ``query`` and return matching memories.

        This is the single public entry point of the retriever. Subclasses
        own the full pipeline behind it: query rewriting, candidate
        generation, filtering, ranking, and packaging the final list.

        Args:
            query: Natural language query for retrieval
            top_k: Maximum number of memories to retrieve (overrides config)
            filters: Optional metadata filters to apply during retrieval
            **kwargs: Additional retrieval-specific parameters

        Returns:
            RetrievalResult containing retrieved memories and metadata
        """
        raise NotImplementedError
