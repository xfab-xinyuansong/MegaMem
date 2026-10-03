from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from megamem.core.memory_entry import MemoryEntry


class BaseMemoryRetriever(ABC):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[MemoryEntry]:
        raise NotImplementedError
