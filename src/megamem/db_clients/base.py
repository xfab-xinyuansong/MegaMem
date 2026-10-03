from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDBClient(ABC):

    @abstractmethod
    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
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
        pass

    @abstractmethod
    def delete(self, collection, ids: List[str]):
        pass

    @abstractmethod
    def count(self, collection) -> int:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass
