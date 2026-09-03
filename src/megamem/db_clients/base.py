"""Abstract interface implemented by every vector-database client."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDBClient(ABC):
    """Backend-agnostic contract for vector store clients."""

    @abstractmethod
    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """Return an existing collection, creating it on first use."""
        pass

    @abstractmethod
    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Insert new records or replace existing ones."""
        pass

    @abstractmethod
    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a similarity (vector) search against the collection."""
        pass

    @abstractmethod
    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fetch records by id and/or metadata filter."""
        pass

    @abstractmethod
    def delete(self, collection, ids: List[str]):
        """Remove records from the collection by id."""
        pass

    @abstractmethod
    def count(self, collection) -> int:
        """Return the number of records in the collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """Drop the entire collection from the backing store."""
        pass
