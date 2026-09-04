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
    """Common base class for all memory-retrieval strategies."""

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
        """Run a retrieval pass for ``query`` and return matching memories."""
        raise NotImplementedError
