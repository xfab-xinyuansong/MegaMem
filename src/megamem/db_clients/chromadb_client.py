from typing import Any, Dict, List, Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Where
from omegaconf import DictConfig, OmegaConf

from megamem.db_clients.base import VectorDBClient
from megamem.utils.embedding import BaseEmbeddingModel


class ChromaDBEmbeddingFunction(EmbeddingFunction):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.embedding_model = BaseEmbeddingModel(cfg)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_model.generate_embeddings(input)

    @staticmethod
    def name() -> str:
        return "external_baseline-general-embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"cfg_yaml": OmegaConf.to_yaml(self.cfg)}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "ChromaDBEmbeddingFunction":
        return cls(OmegaConf.create(config["cfg_yaml"]))


class ChromaDBClient(VectorDBClient):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(path=cfg.memory.persist_path)
        self.embedding_function = ChromaDBEmbeddingFunction(cfg)

    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
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
        return collection.get(
            ids=ids,
            where=where,
            include=include,
            limit=limit,
            offset=offset,
        )

    def delete(self, collection, ids: List[str]):
        collection.delete(ids=ids)

    def count(self, collection) -> int:
        return collection.count()

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)
