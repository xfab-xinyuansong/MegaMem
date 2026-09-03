"""
ChromaDB storage wrapper for document_eval.

Each "kind" of entry lives in its own ChromaDB collection. Collection names
all contain "_doc_" so that LoCoMo (which uses different naming) is physically
isolated.

Kinds:
- raw_chunks (DDI Stream A, also HDM Layer 4, also CDM source backref)
- distilled_memory (DDI Stream B)
- cognitive (CDM)
- section_summaries (HDM Layer 3)
- doc_summaries (HDM Layer 2)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction

from megamem.document_eval.llm_clients import embed_texts
from megamem.document_eval.types import DocumentRetrievalConfig

logger = logging.getLogger(__name__)


class LocalEmbeddingFn(EmbeddingFunction):
    """ChromaDB embedding function backed by our shared local embedder."""

    def __init__(self, cfg: DocumentRetrievalConfig):
        self._cfg = cfg

    def __call__(self, input):
        return embed_texts(self._cfg, list(input))

    @staticmethod
    def name() -> str:
        return "doc_eval_local_st_embedding"


_CLIENT_CACHE: Dict[str, "chromadb.api.client.ClientAPI"] = {}


def _get_client(cfg: DocumentRetrievalConfig):
    key = cfg.chroma_path
    c = _CLIENT_CACHE.get(key)
    if c is None:
        os.makedirs(cfg.chroma_path, exist_ok=True)
        c = chromadb.PersistentClient(path=cfg.chroma_path)
        _CLIENT_CACHE[key] = c
    return c


class DocumentStorage:
    """Wrapper exposing collection-level upsert + query."""

    KINDS = ("raw_chunks", "distilled_memory", "cognitive", "section_summaries", "doc_summaries")

    def __init__(self, cfg: DocumentRetrievalConfig):
        self.cfg = cfg
        self._client = _get_client(cfg)
        self._embed_fn = LocalEmbeddingFn(cfg)
        self._collections: Dict[str, Any] = {}

    def collection(self, kind: str):
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind: {kind}")
        col = self._collections.get(kind)
        if col is None:
            name = self.cfg.collection_name(kind)
            col = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embed_fn,
            )
            self._collections[kind] = col
        return col

    def upsert(
        self,
        kind: str,
        *,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        col = self.collection(kind)
        n_total = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            n_total += len(batch_ids)
        return n_total

    def count(self, kind: str) -> int:
        col = self.collection(kind)
        return col.count()

    def query(
        self,
        kind: str,
        *,
        query_texts: List[str],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        col = self.collection(kind)
        kwargs: Dict[str, Any] = {
            "query_texts": query_texts,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def get_by_ids(self, kind: str, ids: List[str]) -> Dict[str, Any]:
        col = self.collection(kind)
        return col.get(ids=ids, include=["documents", "metadatas"])

    def reset_collection(self, kind: str) -> None:
        """Delete and recreate a single collection (for force_rebuild)."""
        name = self.cfg.collection_name(kind)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collections.pop(kind, None)
        _ = self.collection(kind)

    def reset_all(self) -> None:
        for k in self.KINDS:
            self.reset_collection(k)


# Source registry (chunk_id -> raw text + metadata) is just a wrapper around the
# raw_chunks collection's get_by_ids — no separate registry needed.
