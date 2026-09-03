from __future__ import annotations
from abc import ABC, abstractmethod
from chromadb.api.types import Where
from typing import Any, Dict, List, Optional

from megamem.core.memory_entry import MemoryEntry


class MemoryBase(ABC):
    """
    Abstract interface for an agent's memory layer.

    Contract:
      - The 'key' is the natural-language index phrase the backend will embed.
      - Concrete classes must keep both the original key and value (attribute) in metadata.
      - IDs may be derived deterministically from the key (e.g., a hash) or assigned by the backend.

    Concrete implementations should be idempotent for repeated keys (e.g., upsert semantics).
    """

    @abstractmethod
    def add(self, entry: MemoryEntry) -> str:
        """
        Persist a memory entry.
        Returns a stable record ID (typically hash(key)).
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_key: str,
        k: int = 5,
        where: Any = None,
        include: Optional[List[str]] = None,
    ) -> Any:
        """
        Vector lookup driven by a natural-language query key.
        Returns whatever the backend produces (e.g., a Chroma query dict).
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record by its natural-language key.
        Should map key -> record_id internally (e.g., via hash) and return a dict shaped:
          { "id": <str>, "metadata": <dict or None>, "document": <str or None> }
        Returns None when no record matches.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove the record matching the given natural-language key.
        Should silently no-op when the record does not exist.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Drop every record in the collection.
        Optional; backends that do not support this may raise NotImplementedError.
        """
        raise NotImplementedError


class MemoryStoreBase(ABC):
    """
    Abstract base for memory store backends.

    Establishes the contract for swappable storage layers, e.g. local
    ChromaDB, an HTTP service, or another vector database.

    Memory stores are responsible for:
    - Persisting and reading memory records along with their vector embeddings
    - Supporting similarity search over those embeddings
    - Tracking metadata and document fields
    - Offering CRUD primitives over individual memory records
    """

    @abstractmethod
    def upsert(
        self,
        index: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert (or refresh) a memory record keyed by 'index'.

        Args:
            index: Natural-language phrase that will be embedded
            value: The content to associate with the index
            metadata: Optional supplementary metadata

        Returns:
            Record ID derived from the index
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query: str,
        k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """
        Vector search to surface memories similar to the supplied context.

        Args:
            query: Context string to match against
            k: Number of records to return
            where: Metadata filter conditions
            include: Optional list of fields to include in results

        Returns:
            Backend-specific result object
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, user_id: str) -> MemoryEntry:
        """
        Fetch one record using its natural-language key.

        Args:
            key: Natural-language key to look up

        Returns:
            Dict with id, metadata, document fields — or None when missing
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove a record identified by its natural-language key.

        Args:
            key: Natural-language key to delete
        """
        raise NotImplementedError

    @abstractmethod
    def list_memories(self, limit: int = 10) -> Dict[str, Any]:
        """
        Return memories from the collection.

        Args:
            limit: Maximum number of records to return

        Returns:
            Dict containing the memory records
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Number of records currently stored in the collection.

        Returns:
            Integer record count
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove every record from the collection.
        """
        raise NotImplementedError
