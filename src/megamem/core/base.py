from __future__ import annotations
from abc import ABC, abstractmethod
from chromadb.api.types import Where
from typing import Any, Dict, List, Optional

from megamem.core.memory_entry import MemoryEntry


class MemoryBase(ABC):

    @abstractmethod
    def add(self, entry: MemoryEntry) -> str:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_key: str,
        k: int = 5,
        where: Any = None,
        include: Optional[List[str]] = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, user_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MemoryStoreBase(ABC):

    @abstractmethod
    def upsert(
        self,
        index: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query: str,
        k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, user_id: str) -> MemoryEntry:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_memories(self, limit: int = 10) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
