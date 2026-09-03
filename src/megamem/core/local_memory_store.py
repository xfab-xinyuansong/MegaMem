"""
Local persistent memory store backed by ChromaDB.
"""
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
    """
    Memory store backed by a local ChromaDB PersistentClient.

    Persists memory records to local files. Data survives across sessions
    because ChromaDB writes to the configured persist_path.
    """

    # Class-level locks keyed by user_id so all instances for a user share one lock.
    _user_locks = {}  # Dict[str, threading.RLock] - per-user locks
    _locks_lock = threading.RLock()  # guards mutations of _user_locks

    @classmethod
    def _get_user_lock(cls, user_id: str) -> threading.RLock:
        """
        Return (creating if necessary) the lock for a given user_id.
        Every instance with the same user_id ends up sharing one lock.
        """
        with cls._locks_lock:
            existing = cls._user_locks.get(user_id)
            if existing is None:
                existing = threading.RLock()
                cls._user_locks[user_id] = existing
            return existing

    def __init__(self, cfg: DictConfig, user_id: str):
        """
        Set up the local persistent memory store.

        Args:
            cfg: Configuration object containing memory settings
            user_id: Optional user identifier for collection isolation.
        """
        self.cfg = cfg
        self.user_id = user_id

        persist_path = cfg.memory.persist_path
        distance = cfg.memory.distance

        print("Vector database path:", persist_path)

        # Pick the vector DB backend (ChromaDB by default, Redis if configured) per cfg.memory.db_type.
        self.db_client: VectorDBClient = create_vector_db_client(cfg)
        self.embedding_model = BaseEmbeddingModel(cfg)

        self._embedding_cache = OrderedDict()
        self._cache_max_size = 300

        # Acquire the shared per-user lock (shared with any other instance of this user).
        self._lock = self._get_user_lock(user_id)

        # Resolve the user-specific collection name.
        # For email-shaped user IDs, drop the domain to keep collection names cleaner.
        user_alias = user_id.split('@')[0] if '@' in user_id else user_id
        self.collection_name = f"{cfg.memory.collection_name}_{user_alias}"
        self.collection = self._get_or_create_collection(self.collection_name)

        # BM25 indices keyed by user.
        self._bm25_indices = {}  # user_id -> BM25Okapi index
        self._bm25_doc_ids = {}  # user_id -> list of document IDs

    def _get_or_create_collection(self, collection_name: str):
        """Return the existing collection or create a new one if it isn't there yet."""
        logger.info(f"Getting or creating collection: {collection_name}")
        return self.db_client.get_or_create_collection(
            collection_name=collection_name,
            metadata={"hnsw:space": self.cfg.memory.distance},
        )

    def _get_cached_embedding(self, index: str) -> Optional[List[float]]:
        """Return cached embedding if present, refreshing its position in the LRU."""
        if index not in self._embedding_cache:
            return None
        embedding = self._embedding_cache.pop(index)
        self._embedding_cache[index] = embedding
        return embedding

    def _cache_embedding(self, index: str, embedding: List[float]) -> None:
        """Insert an embedding into the cache, evicting oldest entry when full."""
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
        """
        Insert or update a memory record keyed by 'index'.
        Reuses cached embeddings when possible to avoid recomputation.

        Args:
            index: Natural language index to be embedded
            value: Value/content to store
            metadata: Additional metadata to store

        Returns:
            Record ID (derived from index)
        """
        with self._lock:
            rid = index_to_id(index)

            meta = {"index": index, "value": value}
            if metadata:
                # Serialize image_urls list to JSON if present.
                if "image_urls" in metadata and isinstance(
                    metadata["image_urls"], list
                ):
                    metadata = metadata.copy()  # avoid mutating caller's dict
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
        """
        Vector search to surface memories similar to the supplied context.
        Retries to ride out transient ChromaDB indexing delays.

        Args:
            query: query context string to search for
            top_k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results

        Returns:
            List of MemoryEntry objects
        """
        with self._lock:
            include = include or ["metadatas", "distances"]

        # Retry to absorb vector-database indexing latency.
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
                    retry_delay *= 2  # exponential backoff
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
        """
        Keyword search against both index and value fields.
        Walks keywords from longest to shortest and stops early once we have enough hits.

        Longer keywords (3+ words) also match against contiguous subphrases (so
        "sarah birthday cake" can match "birthday cake"). Shorter keywords (1-2
        words) require an exact phrase match to avoid noisy results.

        Args:
            keywords: List of keywords/phrases to search for
            top_k: Maximum number of results to return
            where: Filter conditions for metadata-based filtering

        Returns:
            List of MemoryEntry objects with scores based on keyword length
        """
        with self._lock:
            # Pull all candidate documents for in-memory phrase matching.
            result = self.db_client.get(
                collection=self.collection,
                where=where,
                include=["metadatas"]
            )

            if not result["metadatas"]:
                return []

        # Lowercase keywords for case-insensitive matching.
        keywords_lower = [kw.lower() for kw in keywords]

        # Track every phrase to search (keywords plus subphrases of long ones), with a score.
        phrase_to_score: Dict[str, int] = {}
        max_word_count = 0

        for keyword in keywords_lower:
            keyword_word_count = len(keyword.split())
            max_word_count = max(max_word_count, keyword_word_count)

            # Original keyword carries its own word-count score.
            if phrase_to_score.get(keyword, -1) < keyword_word_count:
                phrase_to_score[keyword] = keyword_word_count

            # For long keywords, also enumerate contiguous subphrases of size >= 2 (but shorter than the full phrase).
            if keyword_word_count >= 3:
                keyword_words = keyword.split()
                full_len = len(keyword_words)

                for start_idx in range(full_len):
                    for end_idx in range(start_idx + 2, full_len + 1):
                        sub_len = end_idx - start_idx
                        if sub_len >= full_len:
                            continue  # don't duplicate the full phrase
                        subphrase = " ".join(keyword_words[start_idx:end_idx])
                        # Subphrases keep their own word-count score unless we already have a higher one.
                        if phrase_to_score.get(subphrase, -1) < sub_len:
                            phrase_to_score[subphrase] = sub_len

        # Order phrases by word count (then character length) descending so longer / more
        # specific phrases get evaluated first.
        phrases_sorted = sorted(
            phrase_to_score.items(),
            key=lambda pair: (len(pair[0].split()), len(pair[0])),
            reverse=True
        )

        # Track matches keyed by record ID, preserving insertion order.
        matched_docs: "OrderedDict[str, tuple]" = OrderedDict()  # record_id -> (entry, score)

        # Walk phrases longest to shortest.
        for phrase, phrase_score in phrases_sorted:
            phrase_word_count = len(phrase.split())

            # Look for this phrase across the documents.
            for i, metadata in enumerate(result["metadatas"]):
                # Compose the record ID from the index field.
                index = metadata.get("index", "")
                record_id = index_to_id(index)

                # Skip when an existing match has a better-or-equal score.
                if record_id in matched_docs and matched_docs[record_id][1] >= phrase_score:
                    continue

                value = metadata.get("value", "")
                searchable_text = f"{index} {value}".lower()

                # Substring/phrase containment check.
                if phrase in searchable_text:
                    # Insert/replace the entry with this phrase's score when it's higher.
                    if record_id not in matched_docs or matched_docs[record_id][1] < phrase_score:
                        metadata_copy = metadata.copy()
                        metadata_copy["score"] = float(phrase_score)
                        matched_docs[record_id] = (MemoryEntry.from_dict(metadata_copy), phrase_score)

            # Early exit when we already have enough matches and we're moving to short phrases.
            if phrase_word_count <= 2 and len(matched_docs) >= top_k:
                break

        # Rescale scores so they fall below the semantic search threshold.
        # Threshold from config (default 0.4 if absent).
        semantic_threshold = self.cfg.memory.get("query_score_threshold", 0.4)

        # Apply scaling to each matched document.
        for record_id, (entry, raw_score) in matched_docs.items():
            scaled_score = (raw_score / (max_word_count)) * semantic_threshold
            entry.score = scaled_score
            matched_docs[record_id] = (entry, scaled_score)

            # Pull entries out in their existing (priority) insertion order.
            matches = [entry for entry, _ in matched_docs.values()]

            return matches[:top_k]

    def get(self, key: str) -> MemoryEntry:
        """
        Fetch a single record using its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            MemoryEntry, or None when the record doesn't exist
        """
        result = None
        with self._lock:
            record_id = index_to_id(key)

            result = self.db_client.get(
                collection=self.collection,
                ids=[record_id],
                include=["metadatas"]
            )

        # Return None when we got nothing back.
        if not result or not result["ids"]:
            return None

        # Materialize the metadata dict into a MemoryEntry.
        return MemoryEntry.from_dict(result["metadatas"][0])

    def filter(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        Metadata-only filtering — no embedding search involved.

        Wraps ChromaDB's get() with a where clause to surface records that
        match exact metadata constraints (data_type, timestamp_unix ranges, ...).
        Used by the plan executor for FILTER-only retrieval steps.

        Args:
            where: ChromaDB where clause dict for metadata filtering
            limit: Maximum number of results to return

        Returns:
            List of MemoryEntry objects matching the filter conditions
        """
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
        """Remove a record using its natural-language key."""

        record_id = index_to_id(key)
        with self._lock:
            self.db_client.delete(
                collection=self.collection,
                ids=[record_id]
            )

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """Return memory entries from the collection up to *limit*."""
        result = self.db_client.get(
            collection=self.collection,
            include=["documents", "metadatas"],
            limit=limit,
            offset=0
        )

        return [MemoryEntry.from_dict(metadata) for metadata in result["metadatas"]]

    def get_all_cues(self) -> List[MemoryEntry]:
        """Return every cue-index entry (topical + predictive) in the collection.

        Cue entries are identified by having a non-empty ``linked_memory`` field.
        ChromaDB ``get`` with a ``where`` clause fetches metadata without an
        embedding query, making this efficient even for large collections.

        Returns:
            List of cue MemoryEntry objects (each has ``linked_memory`` set).
        """
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
        """Return the number of records stored in the collection."""
        return self.db_client.count(collection=self.collection)

    def clear(self) -> None:
        """Drop every record in the collection and reset caches."""

        # Recreate the underlying collection from scratch.
        with self._lock:
            self.db_client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection(self.collection_name)

        # Drop the embedding cache.
        self._embedding_cache.clear()

        # Drop this user's BM25 index.
        if self.user_id in self._bm25_indices:
            del self._bm25_indices[self.user_id]
        if self.user_id in self._bm25_doc_ids:
            del self._bm25_doc_ids[self.user_id]

    def get_cache_info(self) -> Dict[str, Any]:
        """Snapshot of the embedding cache state."""
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self._cache_max_size,
            "cache_keys": list(self._embedding_cache.keys())[-10:],
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tiny tokenizer for BM25.
        Lowercases the input, replaces common punctuation with spaces, and splits on whitespace.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase first.
        text = text.lower()
        # Translate punctuation into spaces.
        for char in ".,!?;:()[]{}\"'":
            text = text.replace(char, " ")
        # Split and drop blanks.
        return [token for token in text.split() if token]

    def build_bm25_index(self, user_id: Optional[str] = None) -> None:
        """
        Build a BM25 index for a single user from their stored documents.
        Should be invoked after adding memories and before running queries.
        Skipped silently when the collection is empty.

        Args:
            user_id: User ID to build index for. If None, uses self.user_id.

        Note: Only the tokenized corpus and per-document IDs are kept per user. The
        actual metadata is fetched on demand from ChromaDB during search to avoid duplication.
        """
        # Honor the explicit user_id argument when given.
        target_user_id = user_id or self.user_id

        # Bail out fast on an empty collection.
        if self.db_client.count(self.collection) == 0:
            logger.info(f"Collection is empty, skipping BM25 index build for user {target_user_id}")
            return

        logger.info(f"Building BM25 index for user {target_user_id} in collection {self.collection_name}")

        # Pull every document so we can tokenize each one.
        result = self.db_client.get(
            collection=self.collection,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            logger.info(f"No documents found, skipping BM25 index build for user {target_user_id}")
            return

        # Build the tokenized corpus and the parallel doc-IDs list.
        tokenized_corpus = []
        doc_ids = []

        for metadata in result["metadatas"]:
            # Concatenate index and value to form the searchable text.
            index = metadata.get("index", "")
            value = metadata.get("value", "")
            searchable_text = f"{index} {value}"

            # Tokenize, then append to the corpus.
            tokenized_corpus.append(self._tokenize(searchable_text))

            # Remember the corresponding document ID.
            doc_ids.append(index_to_id(index))

        # Build the BM25 index for this user.
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
        """
        BM25 search that ranks documents by relevance for a single user.

        Args:
            query: Query string to search for
            top_k: Maximum number of results to return
            where: Filter conditions for metadata-based filtering (applied post-ranking)
            score_threshold: Minimum BM25 score threshold for filtering results (default: 0.0)

        Returns:
            List of MemoryEntry objects with raw BM25 scores
        """
        # Resolve the target user_id from the where clause first; fall back to self.user_id.
        target_user_id = extract_user_id_from_where(where) or self.user_id

        # Lazily build the BM25 index for this user if missing.
        if target_user_id not in self._bm25_indices:
            logger.info(f"BM25 index not found for user {target_user_id}, building now...")
            self.build_bm25_index(user_id=target_user_id)

            # Bail out if it still isn't built.
            if target_user_id not in self._bm25_indices:
                logger.warning(f"Failed to build BM25 index for user {target_user_id}")
                return []

        # Grab this user's BM25 index plus the doc IDs.
        bm25_index = self._bm25_indices[target_user_id]
        doc_ids = self._bm25_doc_ids[target_user_id]

        # Tokenize the incoming query.
        tokenized_query = self._tokenize(query)

        # BM25 scores across all of this user's documents.
        scores = bm25_index.get_scores(tokenized_query)

        # Pair (doc_id, score) and sort descending.
        doc_scores = [(doc_ids[i], scores[i]) for i in range(len(scores))]
        doc_scores.sort(key=lambda pair: pair[1], reverse=True)

        # Take the top_k candidates that clear the threshold.
        top_doc_ids = [doc_id for doc_id, score in doc_scores[:top_k] if score >= score_threshold]

        if not top_doc_ids:
            return []

        # Fetch metadata for the top documents.
        result = self.db_client.get(
            collection=self.collection,
            ids=top_doc_ids,
            where=where,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            return []

        # Map doc_id -> metadata for fast lookup.
        id_to_metadata = {doc_id: metadata for doc_id, metadata in zip(result["ids"], result["metadatas"])}

        # Walk results in score order, deduplicating linked memories along the way.
        matches = []
        seen_indices: set = set()  # already-added memory indices

        for doc_id, bm25_score in doc_scores:
            # Stop once we've collected enough.
            if len(matches) >= top_k:
                break

            # Skip if the where clause filtered this one out.
            if doc_id not in id_to_metadata:
                continue

            # Use the raw BM25 score for the entry.
            metadata = id_to_metadata[doc_id].copy()
            metadata["score"] = float(bm25_score)

            entry = MemoryEntry.from_dict(metadata)

            # If this entry is a cue index, expand to its primary memories.
            if entry.is_cue_index():
                for primary_index in entry.get_linked_memories():
                    if primary_index in seen_indices:
                        continue  # already added

                    primary_entry = self.get(primary_index)
                    if not primary_entry:
                        continue

                    # Drop episodic memories.
                    if primary_entry.memory_type == "episodic":
                        continue

                    # Carry over the BM25 score from the matching cue.
                    primary_entry.score = float(bm25_score)
                    matches.append(primary_entry)
                    seen_indices.add(primary_index)
            else:
                # Drop episodic memories.
                if entry.memory_type == "episodic":
                    continue

                if entry.index not in seen_indices:
                    matches.append(entry)
                    seen_indices.add(entry.index)

        return matches
