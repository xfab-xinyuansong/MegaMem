from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class TokenCounter:
    def __init__(self, encoding: str = "cl100k_base"):
        self.name = encoding
        self._encoding = None

    @property
    def encoding(self):
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self.name)
        return self._encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


class PersistentIndex:
    def __init__(self, config: dict, build_mode: bool = False):
        self.config = config
        self.path = Path(config["index"]["path"])
        self.build_mode = build_mode
        self._db = None
        self._chroma = None
        self._collections = {}

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def read_manifest(self) -> dict:
        with self.manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest(self, value: dict) -> None:
        if not self.build_mode:
            raise RuntimeError("An inference index cannot write a manifest")
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / "manifest.json.pending"
        temporary.write_text(stable_json(value) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    @property
    def db(self):
        if self._db is None:
            database = self.path / "sources.sqlite3"
            if self.build_mode:
                self.path.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(str(database))
                self._db.executescript(
                    "CREATE TABLE IF NOT EXISTS documents "
                    "(doc_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, position INTEGER NOT NULL, "
                    "token_count INTEGER NOT NULL, state TEXT NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS chunks "
                    "(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, payload TEXT NOT NULL, length INTEGER NOT NULL);"
                    "CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);"
                    "CREATE TABLE IF NOT EXISTS memories "
                    "(memory_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, payload TEXT NOT NULL);"
                    "CREATE INDEX IF NOT EXISTS memories_doc ON memories(doc_id);"
                    "CREATE TABLE IF NOT EXISTS terms "
                    "(term TEXT NOT NULL, chunk_id TEXT NOT NULL, frequency INTEGER NOT NULL, "
                    "PRIMARY KEY(term, chunk_id));"
                    "CREATE INDEX IF NOT EXISTS terms_chunk ON terms(chunk_id);"
                )
            else:
                if not database.is_file():
                    raise FileNotFoundError(f"Missing source store: {database}")
                self._db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            self._db.row_factory = sqlite3.Row
        return self._db

    def collection(self, view: str):
        if view not in {"raw_chunks", "distilled_memory"}:
            raise ValueError(f"Unknown index view: {view}")
        if view not in self._collections:
            if self._chroma is None:
                import chromadb

                location = self.path / "chroma"
                if not self.build_mode and not (location / "chroma.sqlite3").is_file():
                    raise FileNotFoundError(f"Missing persistent Chroma store: {location}")
                self._chroma = chromadb.PersistentClient(path=str(location))
            if self.build_mode:
                collection = self._chroma.get_or_create_collection(
                    name=view, metadata={"hnsw:space": "cosine"}, embedding_function=None
                )
            else:
                collection = self._chroma.get_collection(name=view, embedding_function=None)
            if (collection.metadata or {}).get("hnsw:space") != "cosine":
                raise ValueError(f"Index {view} does not use the configured cosine distance")
            self._collections[view] = collection
        return self._collections[view]

    def document(self, doc_id: str):
        return self.db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()

    def stage_document(self, document: dict, chunks: list[dict], memories: list[dict]) -> None:
        if not self.build_mode:
            raise RuntimeError("An inference index cannot stage documents")
        with self.db:
            self.db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, 'staged')",
                (document["doc_id"], document["fingerprint"], document["position"], document["token_count"]),
            )
            for chunk in chunks:
                frequencies = Counter(lexical_tokens(chunk["content"]))
                self.db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (chunk["chunk_id"], chunk["doc_id"], stable_json(chunk), sum(frequencies.values())),
                )
                self.db.executemany(
                    "INSERT INTO terms VALUES (?, ?, ?)",
                    ((term, chunk["chunk_id"], count) for term, count in frequencies.items()),
                )
            self.db.executemany(
                "INSERT INTO memories VALUES (?, ?, ?)",
                ((memory["memory_id"], memory["doc_id"], stable_json(memory)) for memory in memories),
            )

    def staged_records(self, doc_id: str) -> tuple[list[dict], list[dict]]:
        chunks = [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM chunks WHERE doc_id=? ORDER BY chunk_id", (doc_id,)
        )]
        memories = [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM memories WHERE doc_id=? ORDER BY memory_id", (doc_id,)
        )]
        return chunks, memories

    def finish_document(self, doc_id: str) -> None:
        if not self.build_mode:
            raise RuntimeError("An inference index cannot update documents")
        with self.db:
            self.db.execute("UPDATE documents SET state='ready' WHERE doc_id=?", (doc_id,))

    def counts(self) -> dict:
        return {
            "documents": self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "ready_documents": self.db.execute("SELECT COUNT(*) FROM documents WHERE state='ready'").fetchone()[0],
            "raw_chunks": self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "distilled_memories": self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "source_tokens": self.db.execute("SELECT COALESCE(SUM(token_count),0) FROM documents").fetchone()[0],
        }

    def chunks(self, chunk_ids: Iterable[str]) -> dict[str, dict]:
        identifiers = list(dict.fromkeys(chunk_ids))
        found = {}
        for start in range(0, len(identifiers), 400):
            batch = identifiers[start:start + 400]
            placeholders = ",".join("?" for _ in batch)
            for row in self.db.execute(
                f"SELECT chunk_id, payload FROM chunks WHERE chunk_id IN ({placeholders})", batch
            ):
                found[row[0]] = json.loads(row[1])
        return found

    def query(self, view: str, embedding: list[float], depth: int, scope: list[str] | None = None) -> list[dict]:
        if scope == []:
            return []
        collection = self.collection(view)
        count = collection.count()
        if not count:
            return []
        options = {"where": {"doc_id": {"$in": scope}}} if scope is not None else {}
        response = collection.query(
            query_embeddings=[embedding], n_results=min(depth, count),
            include=["documents", "metadatas", "distances"],
            **options,
        )
        hits = []
        for position, identifier in enumerate(response["ids"][0]):
            metadata = response["metadatas"][0][position]
            sources = json.loads(metadata["source_chunk_ids"])
            hits.append({
                "id": identifier, "content": response["documents"][0][position],
                "source_chunk_ids": sources, "doc_id": metadata["doc_id"],
                "distance": float(response["distances"][0][position]),
                "representation": metadata["representation"],
                "section_path": metadata.get("section_path", ""),
            })
        return hits

    def bm25(self, query: str, depth: int, k1: float, b: float, scope: list[str] | None = None) -> list[dict]:
        if scope == []:
            return []
        terms = list(dict.fromkeys(lexical_tokens(query)))
        if not terms:
            return []
        scope_clause = " WHERE doc_id IN (" + ",".join("?" for _ in scope) + ")" if scope is not None else ""
        total, average = self.db.execute("SELECT COUNT(*), AVG(length) FROM chunks" + scope_clause, scope or []).fetchone()
        if not total or not average:
            return []
        weights = []
        for term in terms:
            if scope is None:
                frequency = self.db.execute("SELECT COUNT(*) FROM terms WHERE term=?", (term,)).fetchone()[0]
            else:
                frequency = self.db.execute(
                    "SELECT COUNT(*) FROM terms t JOIN chunks c ON t.chunk_id=c.chunk_id "
                    "WHERE t.term=? AND c.doc_id IN (" + ",".join("?" for _ in scope) + ")", [term, *scope],
                ).fetchone()[0]
            if frequency:
                weights.append((term, math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))))
        if not weights:
            return []
        values = ",".join("(?,?)" for _ in weights)
        parameters = [item for pair in weights for item in pair]
        parameters.extend([k1, k1, b, b, average])
        parameters.extend(scope or [])
        parameters.append(depth)
        filtered = "WHERE c.doc_id IN (" + ",".join("?" for _ in scope) + ") " if scope is not None else ""
        rows = self.db.execute(
            f"WITH query_terms(term, weight) AS (VALUES {values}) "
            "SELECT c.chunk_id, c.payload, "
            "SUM(q.weight * t.frequency * (? + 1) / "
            "(t.frequency + ? * (1 - ? + ? * c.length / ?))) AS score "
            "FROM query_terms q JOIN terms t ON q.term=t.term "
            "JOIN chunks c ON c.chunk_id=t.chunk_id " + filtered + "GROUP BY c.chunk_id "
            "ORDER BY score DESC, c.chunk_id ASC LIMIT ?", parameters,
        )
        result = []
        for row in rows:
            chunk = json.loads(row[1])
            result.append({
                "id": chunk["chunk_id"], "content": chunk["content"],
                "source_chunk_ids": [chunk["chunk_id"]], "doc_id": chunk["doc_id"],
                "score": row[2], "representation": "raw", "section_path": chunk["section_path"],
            })
        return result

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        self._collections.clear()
        self._chroma = None
