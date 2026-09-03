"""
ChromaDB browser primitives.

Wraps a persistent ChromaDB collection with a thin, dataclass-based API
for listing, searching, filtering and exporting documents.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


@dataclass
class ChromaDocument:
    """A single document materialised from a ChromaDB result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    distance: Optional[float] = None


@dataclass
class ChromaStats:
    """Aggregate metrics describing a ChromaDB collection."""
    total_documents: int
    collection_name: str
    metadata_keys: List[str]
    unique_metadata_values: Dict[str, int]
    content_stats: Dict[str, Any]


class ChromaBrowser:
    """
    Lightweight wrapper around a ChromaDB collection.

    Exposes search, filtering and analysis helpers on top of the underlying
    persistent client.
    """

    def __init__(self, db_path: str, collection_name: str = None):
        """
        Build the browser.

        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to browse (if None, will list available)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Please install with: pip install chromadb")

        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = None
        self.collection = None
        self._initialize_client()

        self._document_cache: List[ChromaDocument] = []
        self._cache_valid = False

    def _initialize_client(self) -> None:
        """Open the persistent ChromaDB client and bind the target collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False),
            )

            available = self.client.list_collections()
            names = [col.name for col in available]

            if not names:
                raise ValueError(f"No collections found in database: {self.db_path}")

            if self.collection_name is None:
                # Default to the first collection if the caller didn't pick one.
                self.collection_name = names[0]
            elif self.collection_name not in names:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. Available: {names}"
                )

            self.collection = self.client.get_collection(self.collection_name)
            self.logger.info(f"Connected to ChromaDB collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def list_collections(self) -> List[str]:
        """
        Enumerate every collection in the persistent store.

        Returns:
            List[str]: Collection names
        """
        try:
            return [col.name for col in self.client.list_collections()]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {str(e)}")
            return []

    def switch_collection(self, collection_name: str) -> bool:
        """
        Re-bind the browser to a different collection.

        Args:
            collection_name: Name of collection to switch to

        Returns:
            bool: True if successful
        """
        try:
            self.collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            self._cache_valid = False  # invalidate any cached docs
            self.logger.info(f"Switched to collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch collection: {str(e)}")
            return False

    def get_all_documents(
        self,
        limit: Optional[int] = None,
        include_embeddings: bool = False,
    ) -> List[ChromaDocument]:
        """
        Return every document in the collection (up to ``limit``).

        Args:
            limit: Maximum number of documents to return
            include_embeddings: Whether to include embedding vectors

        Returns:
            List[ChromaDocument]: Documents from the collection
        """
        try:
            include_list = ["documents", "metadatas"]
            if include_embeddings:
                include_list.append("embeddings")

            payload = self.collection.get(limit=limit, include=include_list)

            ids = payload.get("ids", [])
            contents = payload.get("documents", [])
            metadatas = payload.get("metadatas", [])
            embeddings = payload.get("embeddings", []) if include_embeddings else [None] * len(ids)

            documents: List[ChromaDocument] = []
            for pos, doc_id in enumerate(ids):
                content = contents[pos] if pos < len(contents) else ""
                metadata = metadatas[pos] if pos < len(metadatas) else {}
                embedding = embeddings[pos] if embeddings[pos] is not None else None

                documents.append(
                    ChromaDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        embeddings=embedding,
                    )
                )

            self._document_cache = documents
            self._cache_valid = True

            self.logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to get documents: {str(e)}")
            return []

    def search_documents(
        self,
        query: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ChromaDocument]:
        """
        Run a semantic similarity search.

        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Metadata filter conditions

        Returns:
            List[ChromaDocument]: Matching documents with distances
        """
        try:
            payload = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            documents: List[ChromaDocument] = []
            ids_outer = payload.get("ids") or []
            if ids_outer and len(ids_outer) > 0:
                ids = ids_outer[0]
                contents = payload["documents"][0]
                metadatas = payload["metadatas"][0]
                distances = payload["distances"][0]

                for pos, doc_id in enumerate(ids):
                    content = contents[pos] if pos < len(contents) else ""
                    metadata = metadatas[pos] if pos < len(metadatas) else {}
                    distance = distances[pos] if pos < len(distances) else None

                    documents.append(
                        ChromaDocument(
                            id=doc_id,
                            content=content,
                            metadata=metadata,
                            distance=distance,
                        )
                    )

            self.logger.info(f"Found {len(documents)} matching documents")
            return documents

        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []

    def get_collection_stats(self) -> ChromaStats:
        """
        Compute aggregate statistics over the collection.

        Returns:
            ChromaStats: Collection statistics
        """
        try:
            if not self._cache_valid:
                self.get_all_documents()

            documents = self._document_cache

            metadata_keys: set = set()
            metadata_values: Dict[str, set] = {}
            content_lengths: List[int] = []

            for doc in documents:
                metadata_keys.update(doc.metadata.keys())

                for key, value in doc.metadata.items():
                    metadata_values.setdefault(key, set()).add(str(value))

                content_lengths.append(len(doc.content))

            content_stats: Dict[str, Any] = {}
            if content_lengths:
                total = sum(content_lengths)
                content_stats = {
                    "total_characters": total,
                    "average_length": total / len(content_lengths),
                    "min_length": min(content_lengths),
                    "max_length": max(content_lengths),
                }

            unique_metadata_counts = {key: len(values) for key, values in metadata_values.items()}

            return ChromaStats(
                total_documents=len(documents),
                collection_name=self.collection_name,
                metadata_keys=list(metadata_keys),
                unique_metadata_values=unique_metadata_counts,
                content_stats=content_stats,
            )

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return ChromaStats(
                total_documents=0,
                collection_name=self.collection_name,
                metadata_keys=[],
                unique_metadata_values={},
                content_stats={},
            )

    def filter_documents(
        self,
        where_filter: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[ChromaDocument]:
        """
        Restrict documents to those matching ``where_filter``.

        Args:
            where_filter: Filter conditions
            limit: Maximum results to return

        Returns:
            List[ChromaDocument]: Filtered documents
        """
        try:
            payload = self.collection.get(
                where=where_filter,
                limit=limit,
                include=["documents", "metadatas"],
            )

            ids = payload.get("ids", [])
            contents = payload.get("documents", [])
            metadatas = payload.get("metadatas", [])

            documents: List[ChromaDocument] = []
            for pos, doc_id in enumerate(ids):
                content = contents[pos] if pos < len(contents) else ""
                metadata = metadatas[pos] if pos < len(metadatas) else {}
                documents.append(
                    ChromaDocument(id=doc_id, content=content, metadata=metadata)
                )

            self.logger.info(f"Filtered to {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Filter failed: {str(e)}")
            return []

    def export_documents(
        self,
        documents: List[ChromaDocument],
        file_path: str,
        format: str = "json",
    ) -> None:
        """
        Persist ``documents`` to disk in the requested format.

        Args:
            documents: Documents to export
            file_path: Output file path
            format: Export format (json, csv, txt)
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            handlers = {
                "json": self._export_json,
                "csv": self._export_csv,
                "txt": self._export_txt,
            }
            handler = handlers.get(format.lower())
            if handler is None:
                raise ValueError(f"Unsupported export format: {format}")

            handler(documents, file_path)
            self.logger.info(f"Exported {len(documents)} documents to {file_path}")

        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise

    def _export_json(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Write ``documents`` as a JSON document with light metadata."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "total_documents": len(documents),
            "documents": [asdict(doc) for doc in documents],
        }

        with open(file_path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def _export_csv(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Write ``documents`` to a CSV file with a small per-row preview."""
        import csv

        with open(file_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['id', 'content_length', 'content_preview', 'metadata'])

            for doc in documents:
                preview = doc.content[:100] + '...' if len(doc.content) > 100 else doc.content
                writer.writerow([
                    doc.id,
                    len(doc.content),
                    preview,
                    json.dumps(doc.metadata),
                ])

    def _export_txt(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Write a human-readable plain-text dump of ``documents``."""
        with open(file_path, 'w', encoding='utf-8') as fh:
            fh.write(f"ChromaDB Export - {datetime.now().isoformat()}\n")
            fh.write(f"Collection: {self.collection_name}\n")
            fh.write(f"Total Documents: {len(documents)}\n")
            fh.write("=" * 80 + "\n\n")

            for rank, doc in enumerate(documents, 1):
                truncated = doc.content[:500]
                ellipsis = '...' if len(doc.content) > 500 else ''
                fh.write(f"Document {rank}:\n")
                fh.write(f"  ID: {doc.id}\n")
                fh.write(f"  Length: {len(doc.content)} characters\n")
                fh.write(f"  Metadata: {json.dumps(doc.metadata, indent=4)}\n")
                fh.write(f"  Content: {truncated}{ellipsis}\n")
                fh.write("-" * 40 + "\n\n")
