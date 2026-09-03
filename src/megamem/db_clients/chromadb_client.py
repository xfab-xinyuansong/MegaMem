"""ChromaDB-backed vector store client."""
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Where
from omegaconf import DictConfig, OmegaConf

from megamem.db_clients.base import VectorDBClient
from megamem.utils.embedding import BaseEmbeddingModel


class ChromaDBEmbeddingFunction(EmbeddingFunction):
    """Adapter exposing ``BaseEmbeddingModel`` to ChromaDB."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.embedding_model = BaseEmbeddingModel(cfg)

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents using the wrapped model."""
        return self.embedding_model.generate_embeddings(input)

    @staticmethod
    def name() -> str:
        """Stable identifier consumed by ChromaDB's serialisation layer."""
        return "external_baseline-general-embedding"

    def get_config(self) -> Dict[str, Any]:
        """Return a config blob that can later rebuild this function."""
        return {"cfg_yaml": OmegaConf.to_yaml(self.cfg)}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "ChromaDBEmbeddingFunction":
        """Reconstruct the embedding function from a previously stored config."""
        return cls(OmegaConf.create(config["cfg_yaml"]))


class ChromaDBClient(VectorDBClient):
    """``VectorDBClient`` implementation that uses a ChromaDB persistent store."""

    def __init__(self, cfg: DictConfig):
        """Initialise a ChromaDB client.

        Args:
            cfg: configuration object whose ``memory.persist_path`` controls
                the on-disk location of the database.
        """
        self.cfg = cfg
        self.client = chromadb.PersistentClient(path=cfg.memory.persist_path)
        self.embedding_function = ChromaDBEmbeddingFunction(cfg)

    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """Get-or-create a ChromaDB collection.

        Args:
            collection_name: collection identifier.
            metadata: collection metadata (e.g. distance metric).

        Returns:
            ChromaDB collection handle.
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=self.embedding_function,
        )

    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Upsert records into a ChromaDB collection.

        Args:
            collection: ChromaDB collection handle.
            ids: per-record identifiers.
            documents: raw document texts.
            metadatas: per-record metadata dicts.
            embeddings: optional pre-computed embedding vectors.
        """
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a similarity query against a ChromaDB collection.

        Args:
            collection: ChromaDB collection handle.
            query_texts: search query text.
            n_results: maximum number of matches.
            where: optional filter clause.
            include: fields to return.

        Returns:
            ChromaDB query result dictionary.
        """
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            include=include,
        )

    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read records back from a ChromaDB collection.

        Args:
            collection: ChromaDB collection handle.
            ids: optional whitelist of record ids.
            where: optional filter clause.
            include: fields to return.
            limit: maximum number of returned records.
            offset: skip this many records before returning.

        Returns:
            ChromaDB result dictionary.
        """
        return collection.get(
            ids=ids,
            where=where,
            include=include,
            limit=limit,
            offset=offset,
        )

    def delete(self, collection, ids: List[str]):
        """Remove records by id.

        Args:
            collection: ChromaDB collection handle.
            ids: identifiers to delete.
        """
        collection.delete(ids=ids)

    def count(self, collection) -> int:
        """Return the number of records currently in the collection.

        Args:
            collection: ChromaDB collection handle.
        """
        return collection.count()

    def delete_collection(self, collection_name: str):
        """Delete the named collection entirely.

        Args:
            collection_name: collection identifier.
        """
        self.client.delete_collection(collection_name)
