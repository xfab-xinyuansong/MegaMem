"""Redis Stack-backed vector store client.

Requires ``redis-py`` 4.0+. Install with::

    pip install redis
"""
import json
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

# Soft-import the Redis stack so the rest of the package can still be
# imported on environments where ``redis`` is unavailable.
REDIS_AVAILABLE = False
try:
    import numpy as np
    from redis import Redis
    from redis.commands.search.field import NumericField, TextField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    REDIS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

from megamem.db_clients.base import VectorDBClient
from megamem.utils.embedding import BaseEmbeddingModel


class RedisVectorDBClient(VectorDBClient):
    """``VectorDBClient`` backed by Redis Stack search/JSON modules."""

    # Metadata fields that are auto-indexed both as TEXT (for free-text
    # search) and as TAG (for exact filtering). Adding a field here makes
    # it filterable by ``where`` clauses without code changes.
    DEFAULT_INDEXED_FIELDS = [
        "user_id",
        "linked_memory",
        "creation_time",
        "cue_indices",
    ]

    def __init__(self, cfg: DictConfig):
        """Initialise a Redis-backed vector client.

        Args:
            cfg: configuration object; reads connection params and
                ``embedding_dim``/``distance`` from ``cfg.memory``.
        """
        self.cfg = cfg

        host = cfg.memory.get("redis_host", "localhost")
        port = cfg.memory.get("redis_port", 6379)
        db = cfg.memory.get("redis_db", 0)
        password = cfg.memory.get("redis_password", None)

        # We deliberately keep raw bytes from Redis to make vector handling
        # explicit; JSON encoding/decoding is done manually where needed.
        self.client = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,
        )

        self.embedding_model = BaseEmbeddingModel(cfg)

        self.embedding_dim = cfg.memory.get("embedding_dim", 1536)

        metric = cfg.memory.get("distance", "cosine").upper()
        if metric == "COSINE":
            self.distance_metric = "COSINE"
        elif metric == "L2":
            self.distance_metric = "L2"
        else:
            self.distance_metric = "IP"  # Inner product

        self.indexed_fields = self.DEFAULT_INDEXED_FIELDS

        # Cached metadata per collection name.
        self._collections = {}

    def _get_collection_prefix(self, collection_name: str) -> str:
        """Redis-key prefix for documents belonging to ``collection_name``."""
        return f"collection:{collection_name}"

    def _get_index_name(self, collection_name: str) -> str:
        """Search-index name associated with ``collection_name``."""
        return f"idx:{collection_name}"

    def _get_doc_key(self, collection_name: str, doc_id: str) -> str:
        """Full Redis key for a stored document JSON."""
        return f"{self._get_collection_prefix(collection_name)}:doc:{doc_id}"

    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """Return cached collection info, creating the search index on demand.

        Args:
            collection_name: collection identifier.
            metadata: collection metadata stored alongside the cached entry.

        Returns:
            Collection info dictionary used by other methods.
        """
        index_name = self._get_index_name(collection_name)

        try:
            self.client.ft(index_name).info()
            self._collections[collection_name] = {
                "name": collection_name,
                "index_name": index_name,
                "metadata": metadata,
            }
            return self._collections[collection_name]
        except Exception:
            # Index missing — fall through and create it.
            pass

        from redis.commands.search.field import TagField

        # Static schema columns shared by every collection.
        schema_fields = [
            TextField("$.id", as_name="id"),
            TextField("$.document", as_name="document"),
            TextField("$.index", as_name="index"),
            TextField("$.value", as_name="value"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": self.distance_metric,
                },
                as_name="embedding",
            ),
            NumericField("$.timestamp", as_name="timestamp"),
        ]

        # Each indexed metadata field gets both a TEXT and a TAG variant so
        # both fuzzy search and exact filtering are cheap.
        for field_name in self.indexed_fields:
            schema_fields.append(TextField(f"$.{field_name}", as_name=field_name))
            schema_fields.append(
                TagField(f"$.{field_name}", as_name=f"{field_name}_tag")
            )

        schema = tuple(schema_fields)

        definition = IndexDefinition(
            prefix=[f"{self._get_collection_prefix(collection_name)}:doc:"],
            index_type=IndexType.JSON,
        )

        self.client.ft(index_name).create_index(
            fields=schema,
            definition=definition,
        )

        # Re-write any documents that already existed under this prefix so
        # the new index picks them up. This handles the case where the
        # index was previously dropped.
        prefix = self._get_collection_prefix(collection_name)
        existing_keys = self.client.keys(f"{prefix}:doc:*")
        if existing_keys:
            import logging

            log = logging.getLogger(__name__)
            log.info(
                f"Re-indexing {len(existing_keys)} existing documents for collection {collection_name}"
            )
            for key in existing_keys:
                doc = self.client.json().get(key)
                if doc:
                    self.client.json().set(key, "$", doc)

        self._collections[collection_name] = {
            "name": collection_name,
            "index_name": index_name,
            "metadata": metadata,
        }

        return self._collections[collection_name]

    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Insert or replace documents in the Redis collection.

        Args:
            collection: collection info dict from ``get_or_create_collection``.
            ids: per-document identifiers.
            documents: raw document texts.
            metadatas: per-document metadata.
            embeddings: optional pre-computed embedding vectors.
        """
        collection_name = collection["name"]

        if embeddings is None:
            embeddings = self.embedding_model.generate_embeddings(documents)

        for doc_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            doc_key = self._get_doc_key(collection_name, doc_id)

            embedding_list = np.array(embedding, dtype=np.float32).tolist()

            # Coerce timestamp to a number so the NumericField indexes it.
            ts_val = metadata.get("timestamp", 0)
            if isinstance(ts_val, str):
                try:
                    ts_val = float(ts_val) if ts_val else 0
                except Exception:
                    ts_val = 0

            # Redis TAG fields cannot index empty strings, so we substitute
            # a sentinel value and translate it back on read.
            def tag_value(val):
                return "__EMPTY__" if val == "" or val is None else val

            doc_data = {
                "id": doc_id,
                "document": document,
                "index": metadata.get("index", ""),
                "value": metadata.get("value", ""),
                "embedding": embedding_list,
                "timestamp": ts_val,
            }

            for key, val in metadata.items():
                if key in doc_data or key == "timestamp":
                    continue
                if isinstance(val, str):
                    doc_data[key] = tag_value(val)
                else:
                    doc_data[key] = val

            self.client.json().set(doc_key, "$", doc_data)

    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a vector similarity search.

        Args:
            collection: collection info dict from ``get_or_create_collection``.
            query_texts: search query string.
            n_results: max number of results to return.
            where: optional filter (ChromaDB-flavoured ``$and``/``$or``/``$eq`` etc.).
            include: fields to include in results (currently informational).

        Returns:
            Result dictionary in ChromaDB-compatible shape.
        """
        index_name = collection["index_name"]
        collection_name = collection["name"]

        query_embedding = self.embedding_model.generate_embeddings([query_texts])[0]
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        filter_parts: List[str] = []
        post_filters: List[tuple] = []  # Conditions Redis can't natively express.

        def escape_redis_value(val: str) -> str:
            """Escape characters that would otherwise break a TAG query."""
            # Redis TAG values reserve a fairly broad set of punctuation;
            # we conservatively escape every character that has been
            # observed to cause syntax errors in practice.
            special_chars = [
                ",", ".", "<", ">", "{", "}", "[", "]", '"', "'", ":", ";",
                "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+",
                "=", "~", "|", " ",
            ]
            escaped = val
            for ch in special_chars:
                escaped = escaped.replace(ch, f"\\{ch}")
            return escaped

        def process_condition(key, value):
            """Translate a single ``key, value`` clause into Redis filters."""
            if isinstance(value, dict):
                if "$ne" in value:
                    ne_val = value["$ne"]
                    if ne_val == "":
                        # "Has any value" — match any populated TAG entry.
                        filter_parts.append(f"@{key}_tag:{{*}}")
                    else:
                        # Negation against a specific TAG value.
                        escaped_val = escape_redis_value(str(ne_val))
                        filter_parts.append(f"-@{key}_tag:{{{escaped_val}}}")
                elif "$eq" in value:
                    eq_val = value["$eq"]
                    if eq_val == "":
                        filter_parts.append(f"@{key}_tag:{{__EMPTY__}}")
                    else:
                        escaped_val = escape_redis_value(str(eq_val))
                        filter_parts.append(f"@{key}_tag:{{{escaped_val}}}")
                elif "$gt" in value:
                    filter_parts.append(f"@{key}:[({value['$gt']} +inf]")
                elif "$gte" in value:
                    filter_parts.append(f"@{key}:[{value['$gte']} +inf]")
                elif "$lt" in value:
                    filter_parts.append(f"@{key}:[-inf ({value['$lt']}]")
                elif "$lte" in value:
                    filter_parts.append(f"@{key}:[-inf {value['$lte']}]")
            elif isinstance(value, str):
                escaped_val = escape_redis_value(value)
                filter_parts.append(f"@{key}_tag:{{{escaped_val}}}")
            elif isinstance(value, (int, float)):
                filter_parts.append(f"@{key}:[{value} {value}]")
            else:
                filter_parts.append(f"@{key}_tag:{{{str(value)}}}")

        if where:
            if "$and" in where:
                for cond in where["$and"]:
                    for key, value in cond.items():
                        process_condition(key, value)
            elif "$or" in where:
                # Redis OR — combine alternatives with the pipe operator.
                or_parts: List[str] = []
                for cond in where["$or"]:
                    for key, value in cond.items():
                        if isinstance(value, dict):
                            if "$eq" in value and value["$eq"] != "":
                                escaped_val = escape_redis_value(str(value["$eq"]))
                                or_parts.append(f"@{key}_tag:{{{escaped_val}}}")
                            elif "$ne" in value and value["$ne"] == "":
                                or_parts.append(f"@{key}_tag:{{*}}")
                            else:
                                # Anything else is too complex for Redis; defer.
                                for op, op_val in value.items():
                                    post_filters.append((key, op, op_val))
                        elif isinstance(value, str):
                            escaped_val = escape_redis_value(value)
                            or_parts.append(f"@{key}_tag:{{{escaped_val}}}")
                if or_parts:
                    filter_parts.append("(" + "|".join(or_parts) + ")")
            else:
                for key, value in where.items():
                    process_condition(key, value)

        # Decide how many results to ask Redis for, given the mix of
        # native and post filters.
        if filter_parts:
            filter_str = "(" + " ".join(filter_parts) + ")"
            fetch_count = n_results * 2 if post_filters else n_results
            query_str = f"{filter_str}=>[KNN {fetch_count} @embedding $vector AS score]"
        else:
            fetch_count = n_results * 5 if post_filters else n_results
            query_str = f"*=>[KNN {fetch_count} @embedding $vector AS score]"

        return_fields = (
            ["id", "document", "index", "value", "score", "timestamp"]
            + self.indexed_fields
        )

        query = (
            Query(query_str)
            .sort_by("score")
            .return_fields(*return_fields)
            .dialect(2)
        )

        results = self.client.ft(index_name).search(
            query,
            query_params={"vector": query_vector},
        )

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        distances: List[float] = []
        documents: List[str] = []

        for doc in results.docs:
            doc_id = doc.id.split(":")[-1]

            # Re-fetch the full JSON document; search results sometimes
            # omit fields we care about.
            doc_key = self._get_doc_key(collection_name, doc_id)
            full_doc = self.client.json().get(doc_key)

            if not full_doc:
                continue

            timestamp = full_doc.get("timestamp", 0)
            timestamp_str = str(timestamp) if timestamp else ""

            def from_tag_value(val):
                return "" if val == "__EMPTY__" else (val or "")

            metadata = {
                "index": full_doc.get("index", ""),
                "value": full_doc.get("value", ""),
                "timestamp": timestamp_str,
            }

            for key, value in full_doc.items():
                if key in ("id", "document", "embedding", "index", "value", "timestamp"):
                    continue
                if isinstance(value, str):
                    metadata[key] = from_tag_value(value)
                else:
                    metadata[key] = value

            skip = False
            for filter_key, filter_op, filter_val in post_filters:
                field_value = metadata.get(filter_key, "")
                if filter_op == "$ne" and field_value == filter_val:
                    skip = True
                    break
                if filter_op == "$eq" and field_value != filter_val:
                    skip = True
                    break

            if skip:
                continue

            ids.append(doc_id)
            metadatas.append(metadata)

            score = float(getattr(doc, "score", 0))
            # Both COSINE and L2/IP are returned with "smaller is better"
            # semantics in this codepath, so we forward the raw score.
            distances.append(score)

            documents.append(full_doc.get("document", ""))

            if len(ids) >= n_results:
                break

        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
            "documents": [documents],
        }

    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read records back from a Redis collection.

        Args:
            collection: collection info dict from ``get_or_create_collection``.
            ids: optional whitelist of record ids.
            where: optional filter clause.
            include: fields to include in results (informational).
            limit: maximum number of records to return.
            offset: pagination offset (used with ``limit``).

        Returns:
            Result dictionary in ChromaDB-compatible shape.
        """
        collection_name = collection["name"]
        index_name = collection["index_name"]

        def from_tag_value(val):
            return "" if val == "__EMPTY__" else (val or "")

        result_ids: List[str] = []
        result_metadatas: List[Dict[str, Any]] = []
        result_documents: List[str] = []

        if ids:
            for doc_id in ids:
                doc_key = self._get_doc_key(collection_name, doc_id)
                doc_data = self.client.json().get(doc_key)

                if not doc_data:
                    continue

                result_ids.append(doc_id)

                timestamp = doc_data.get("timestamp", 0)
                timestamp_str = str(timestamp) if timestamp else ""

                metadata = {
                    "index": doc_data.get("index", ""),
                    "value": doc_data.get("value", ""),
                    "timestamp": timestamp_str,
                }
                for key, value in doc_data.items():
                    if key in ("id", "document", "embedding", "index", "value", "timestamp"):
                        continue
                    if isinstance(value, str):
                        metadata[key] = from_tag_value(value)
                    else:
                        metadata[key] = value

                result_metadatas.append(metadata)
                result_documents.append(doc_data.get("document", ""))
        else:
            filter_str = "*"
            if where:
                pieces = []
                for key, value in where.items():
                    if isinstance(value, str):
                        pieces.append(f"@{key}:{{{value}}}")
                    else:
                        pieces.append(f"@{key}:[{value} {value}]")
                if pieces:
                    filter_str = " ".join(pieces)

            return_fields = (
                ["id", "index", "value", "document", "timestamp"]
                + self.indexed_fields
            )
            query = Query(filter_str).return_fields(*return_fields)

            if limit:
                query = query.paging(offset or 0, limit)

            results = self.client.ft(index_name).search(query)

            for doc in results.docs:
                doc_id = doc.id.split(":")[-1]
                result_ids.append(doc_id)

                doc_key = self._get_doc_key(collection_name, doc_id)
                doc_data = self.client.json().get(doc_key)

                if not doc_data:
                    continue

                timestamp = doc_data.get("timestamp", 0)
                timestamp_str = str(timestamp) if timestamp else ""

                metadata = {
                    "index": doc_data.get("index", ""),
                    "value": doc_data.get("value", ""),
                    "timestamp": timestamp_str,
                }
                for key, value in doc_data.items():
                    if key in ("id", "document", "embedding", "index", "value", "timestamp"):
                        continue
                    if isinstance(value, str):
                        metadata[key] = from_tag_value(value)
                    else:
                        metadata[key] = value

                result_metadatas.append(metadata)
                result_documents.append(doc_data.get("document", ""))

        return {
            "ids": result_ids,
            "metadatas": result_metadatas,
            "documents": result_documents,
        }

    def delete(self, collection, ids: List[str]):
        """Delete documents by id.

        Args:
            collection: collection info dict.
            ids: list of identifiers to remove.
        """
        collection_name = collection["name"]

        for doc_id in ids:
            self.client.delete(self._get_doc_key(collection_name, doc_id))

    def count(self, collection) -> int:
        """Return the number of documents in the collection.

        Args:
            collection: collection info dict.
        """
        index_name = collection["index_name"]

        try:
            results = self.client.ft(index_name).search(Query("*").paging(0, 0))
            return results.total
        except Exception:
            return 0

    def delete_collection(self, collection_name: str):
        """Drop the index and every document for the given collection.

        Args:
            collection_name: collection identifier to delete.
        """
        index_name = self._get_index_name(collection_name)
        prefix = f"{self._get_collection_prefix(collection_name)}:doc:*"

        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass

        # Sweep up any remaining keys that match the collection prefix.
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=prefix, count=100)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break

        if collection_name in self._collections:
            del self._collections[collection_name]
