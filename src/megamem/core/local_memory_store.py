import logging

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, TypeVar, Union
import time
import threading

from omegaconf import DictConfig
from chromadb.api.types import Where
from rank_bm25 import BM25Okapi

from megamem.core.base import MemoryStoreBase
from megamem.core.memory_entry import MemoryEntry
from megamem.db_clients import VectorDBClient, create_vector_db_client
from megamem.utils.embedding import BaseEmbeddingModel
from megamem.utils.misc import index_to_id, extract_user_id_from_where

logger = logging.getLogger(__name__)

T = TypeVar("T")
OneOrMany = Union[T, List[T]]


class LocalMemoryStore(MemoryStoreBase):

    _user_locks = {}
    _locks_lock = threading.RLock()

    @classmethod
    def _get_user_lock(cls, user_id: str) -> threading.RLock:
        with cls._locks_lock:
            existing = cls._user_locks.get(user_id)
            if existing is None:
                existing = threading.RLock()
                cls._user_locks[user_id] = existing
            return existing

    def __init__(self, cfg: DictConfig, user_id: str):
        self.cfg = cfg
        self.user_id = user_id

        persist_path = cfg.memory.persist_path
        distance = cfg.memory.distance

        print("Vector database path:", persist_path)

        self.db_client: VectorDBClient = create_vector_db_client(cfg)
        self.embedding_model = BaseEmbeddingModel(cfg)

        self._embedding_cache = OrderedDict()
        self._cache_max_size = 300

        self._lock = self._get_user_lock(user_id)

        user_alias = user_id.split('@')[0] if '@' in user_id else user_id
        self.collection_name = f"{cfg.memory.collection_name}_{user_alias}"
        self.collection = self._get_or_create_collection(self.collection_name)

        self._bm25_indices = {}
        self._bm25_doc_ids = {}

    def _get_or_create_collection(self, collection_name: str):
        logger.info(f"Getting or creating collection: {collection_name}")
        return self.db_client.get_or_create_collection(
            collection_name=collection_name,
            metadata={"hnsw:space": self.cfg.memory.distance},
        )

    def _get_cached_embedding(self, index: str) -> Optional[List[float]]:
        if index not in self._embedding_cache:
            return None
        embedding = self._embedding_cache.pop(index)
        self._embedding_cache[index] = embedding
        return embedding

    def _cache_embedding(self, index: str, embedding: List[float]) -> None:
        if index in self._embedding_cache:
            del self._embedding_cache[index]

        self._embedding_cache[index] = embedding

        if len(self._embedding_cache) > self._cache_max_size:
            self._embedding_cache.popitem(last=False)

    def upsert(
        self,
        index: str,
        value: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            rid = index_to_id(index)

            meta = {"index": index, "value": value}
            if metadata:
                if "image_urls" in metadata and isinstance(
                    metadata["image_urls"], list
                ):
                    metadata = metadata.copy()
                    metadata["image_urls"] = json.dumps(metadata["image_urls"])
                meta = {**meta, **metadata}

            cached_embedding = self._get_cached_embedding(index)

            if cached_embedding is not None:
                self.db_client.upsert(
                    collection=self.collection,
                    ids=[rid],
                    documents=[index],
                    metadatas=[meta],
                    embeddings=[cached_embedding],
                )
            else:
                self.db_client.upsert(
                    collection=self.collection,
                    ids=[rid],
                    documents=[index],
                    metadatas=[meta],
                )

            return rid

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            include = include or ["metadatas", "distances"]

        max_retries = 3
        retry_delay = 0.1
        entries: List[MemoryEntry] = []
        for attempt in range(max_retries):
            try:
                result = self.db_client.query(
                    collection=self.collection,
                    query_texts=query,
                    n_results=top_k,
                    where=where,
                    include=include,
                )

                for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
                    metadata["score"] = 1 - distance
                    entries.append(MemoryEntry.from_dict(metadata))
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Query attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s: {str(e)[:100]}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise
        return entries

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 10,
        where: Optional[Where] = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            result = self.db_client.get(
                collection=self.collection,
                where=where,
                include=["metadatas"]
            )

            if not result["metadatas"]:
                return []

        keywords_lower = [kw.lower() for kw in keywords]

        phrase_to_score: Dict[str, int] = {}
        max_word_count = 0

        for keyword in keywords_lower:
            keyword_word_count = len(keyword.split())
            max_word_count = max(max_word_count, keyword_word_count)

            if phrase_to_score.get(keyword, -1) < keyword_word_count:
                phrase_to_score[keyword] = keyword_word_count

            if keyword_word_count >= 3:
                keyword_words = keyword.split()
                full_len = len(keyword_words)

                for start_idx in range(full_len):
                    for end_idx in range(start_idx + 2, full_len + 1):
                        sub_len = end_idx - start_idx
                        if sub_len >= full_len:
                            continue
                        subphrase = " ".join(keyword_words[start_idx:end_idx])
                        if phrase_to_score.get(subphrase, -1) < sub_len:
                            phrase_to_score[subphrase] = sub_len

        phrases_sorted = sorted(
            phrase_to_score.items(),
            key=lambda pair: (len(pair[0].split()), len(pair[0])),
            reverse=True
        )

        matched_docs: "OrderedDict[str, tuple]" = OrderedDict()

        for phrase, phrase_score in phrases_sorted:
            phrase_word_count = len(phrase.split())

            for i, metadata in enumerate(result["metadatas"]):
                index = metadata.get("index", "")
                record_id = index_to_id(index)

                if record_id in matched_docs and matched_docs[record_id][1] >= phrase_score:
                    continue

                value = metadata.get("value", "")
                searchable_text = f"{index} {value}".lower()

                if phrase in searchable_text:
                    if record_id not in matched_docs or matched_docs[record_id][1] < phrase_score:
                        metadata_copy = metadata.copy()
                        metadata_copy["score"] = float(phrase_score)
                        matched_docs[record_id] = (MemoryEntry.from_dict(metadata_copy), phrase_score)

            if phrase_word_count <= 2 and len(matched_docs) >= top_k:
                break

        semantic_threshold = self.cfg.memory.get("query_score_threshold", 0.4)

        for record_id, (entry, raw_score) in matched_docs.items():
            scaled_score = (raw_score / (max_word_count)) * semantic_threshold
            entry.score = scaled_score
            matched_docs[record_id] = (entry, scaled_score)

            matches = [entry for entry, _ in matched_docs.values()]

            return matches[:top_k]

    def get(self, key: str) -> MemoryEntry:
        result = None
        with self._lock:
            record_id = index_to_id(key)

            result = self.db_client.get(
                collection=self.collection,
                ids=[record_id],
                include=["metadatas"]
            )

        if not result or not result["ids"]:
            return None

        return MemoryEntry.from_dict(result["metadatas"][0])

    def filter(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        with self._lock:
            result = self.db_client.get(
                collection=self.collection,
                where=where,
                include=["metadatas"],
                limit=limit,
            )

        if not result or not result["metadatas"]:
            return []

        return [MemoryEntry.from_dict(metadata) for metadata in result["metadatas"]]

    def delete(self, key: str) -> None:

        record_id = index_to_id(key)
        with self._lock:
            self.db_client.delete(
                collection=self.collection,
                ids=[record_id]
            )

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        result = self.db_client.get(
            collection=self.collection,
            include=["documents", "metadatas"],
            limit=limit,
            offset=0
        )

        return [MemoryEntry.from_dict(metadata) for metadata in result["metadatas"]]

    def get_all_cues(self) -> List[MemoryEntry]:
        with self._lock:
            result = self.db_client.get(
                collection=self.collection,
                where={"linked_memory": {"$ne": ""}},
                include=["documents", "metadatas"],
            )

        if not result or not result["metadatas"]:
            return []

        return [MemoryEntry.from_dict(metadata) for metadata in result["metadatas"]]

    def count(self) -> int:
        return self.db_client.count(collection=self.collection)

    def clear(self) -> None:

        with self._lock:
            self.db_client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection(self.collection_name)

        self._embedding_cache.clear()

        if self.user_id in self._bm25_indices:
            del self._bm25_indices[self.user_id]
        if self.user_id in self._bm25_doc_ids:
            del self._bm25_doc_ids[self.user_id]

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self._cache_max_size,
            "cache_keys": list(self._embedding_cache.keys())[-10:],
        }

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for char in ".,!?;:()[]{}\"'":
            text = text.replace(char, " ")
        return [token for token in text.split() if token]

    def build_bm25_index(self, user_id: Optional[str] = None) -> None:
        target_user_id = user_id or self.user_id

        if self.db_client.count(self.collection) == 0:
            logger.info(f"Collection is empty, skipping BM25 index build for user {target_user_id}")
            return

        logger.info(f"Building BM25 index for user {target_user_id} in collection {self.collection_name}")

        result = self.db_client.get(
            collection=self.collection,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            logger.info(f"No documents found, skipping BM25 index build for user {target_user_id}")
            return

        tokenized_corpus = []
        doc_ids = []

        for metadata in result["metadatas"]:
            index = metadata.get("index", "")
            value = metadata.get("value", "")
            searchable_text = f"{index} {value}"

            tokenized_corpus.append(self._tokenize(searchable_text))

            doc_ids.append(index_to_id(index))

        self._bm25_indices[target_user_id] = BM25Okapi(tokenized_corpus)
        self._bm25_doc_ids[target_user_id] = doc_ids

        logger.info(f"BM25 index built for user {target_user_id} with {len(tokenized_corpus)} documents")

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Where] = None,
        score_threshold: float = 0.0,
    ) -> List[MemoryEntry]:
        target_user_id = extract_user_id_from_where(where) or self.user_id

        if target_user_id not in self._bm25_indices:
            logger.info(f"BM25 index not found for user {target_user_id}, building now...")
            self.build_bm25_index(user_id=target_user_id)

            if target_user_id not in self._bm25_indices:
                logger.warning(f"Failed to build BM25 index for user {target_user_id}")
                return []

        bm25_index = self._bm25_indices[target_user_id]
        doc_ids = self._bm25_doc_ids[target_user_id]

        tokenized_query = self._tokenize(query)

        scores = bm25_index.get_scores(tokenized_query)

        doc_scores = [(doc_ids[i], scores[i]) for i in range(len(scores))]
        doc_scores.sort(key=lambda pair: pair[1], reverse=True)

        top_doc_ids = [doc_id for doc_id, score in doc_scores[:top_k] if score >= score_threshold]

        if not top_doc_ids:
            return []

        result = self.db_client.get(
            collection=self.collection,
            ids=top_doc_ids,
            where=where,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            return []

        id_to_metadata = {doc_id: metadata for doc_id, metadata in zip(result["ids"], result["metadatas"])}

        matches = []
        seen_indices: set = set()

        for doc_id, bm25_score in doc_scores:
            if len(matches) >= top_k:
                break

            if doc_id not in id_to_metadata:
                continue

            metadata = id_to_metadata[doc_id].copy()
            metadata["score"] = float(bm25_score)

            entry = MemoryEntry.from_dict(metadata)

            if entry.is_cue_index():
                for primary_index in entry.get_linked_memories():
                    if primary_index in seen_indices:
                        continue

                    primary_entry = self.get(primary_index)
                    if not primary_entry:
                        continue

                    if primary_entry.memory_type == "episodic":
                        continue

                    primary_entry.score = float(bm25_score)
                    matches.append(primary_entry)
                    seen_indices.add(primary_index)
            else:
                if entry.memory_type == "episodic":
                    continue

                if entry.index not in seen_indices:
                    matches.append(entry)
                    seen_indices.add(entry.index)

        return matches
