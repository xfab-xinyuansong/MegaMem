"""
Memory Entry data model.

Defines :class:`MemoryEntry`, the canonical return type for the megamem
query and build pipelines.
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class MemoryMetadata(BaseModel):
    """
    Metadata attached to a memory entry.

    Holds every metadata-style attribute that may travel alongside a
    memory record, including search context, indexing details, timestamps,
    and link information.
    """
    pass


# Metadata keys that map directly to dedicated MemoryEntry attributes.
# Anything else coming back from ChromaDB metadata is funneled into extra_metadata.
_KNOWN_MEMORY_FIELDS = {
    "value", "index", "history", "memory_type", "episodic_memory_ids",
    "score", "timestamp", "query", "creation_time", "linked_memory",
    "cue_indices", "predictive_cue_indices", "image_urls", "data_type",
    "timestamp_unix", "extra_metadata", "cue_type",
}


class MemoryEntry(BaseModel):
    """Canonical memory entry shared by query and build pipelines."""

    # Core memory payload
    value: str = Field(
        description="The main memory content/text"
    )

    index: Optional[str] = Field(
        default="",
        description="Memory index/key for retrieval and identification"
    )

    # Versioning / history
    history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Historical versions of this memory entry"
    )

    # Memory taxonomy
    memory_type: Optional[str] = Field(
        default="",
        description="Type of memory (e.g., 'factual', 'procedural', 'episodic')"
    )

    # Pointer to the originating episodic memories (factual entries)
    episodic_memory_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="List of episodic memory IDs that provide context for this factual memory"
    )

    # Relevance / scoring
    score: Optional[float] = Field(
        default=0.0,
        description="Relevance/similarity score"
    )

    # When in time the memory event occurred
    timestamp: Optional[str] = Field(
        default="",
        description="Timestamp when the memory event occurred"
    )

    # Originating query text
    query: Optional[str] = Field(
        default="",
        description="The original query that retrieved this memory"
    )

    # When the memory record itself was made
    creation_time: Optional[str] = Field(
        default="",
        description="Timestamp when the memory was originally created"
    )

    # Index that this entry refers to (used for indexing-only entries)
    linked_memory: Optional[str] = Field(
        default="",
        description="Reference to linked memory entries"
    )

    # Cue indices when the memory entry connects to multiple indices
    cue_indices: Optional[str] = Field(
        default="", description="Indices linked to this memory for enhanced retrieval"
    )

    # Predictive (extrinsic) cue indices — commonsense-derived phrases that capture
    # likely retrieval contexts; kept distinct from intrinsic cues.
    predictive_cue_indices: Optional[str] = Field(
        default="",
        description="Extrinsic cue indices for bridging semantically distant queries",
    )

    # Image URLs related to this memory
    image_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="List of image URLs associated with this memory"
    )

    # High-level source bucket used for metadata filtering (e.g., "mail", "doc")
    data_type: Optional[str] = Field(
        default="",
        description="Source type category: 'mail', 'doc', or '' (unspecified)"
    )

    # Unix timestamp (seconds) supporting numeric range filtering in ChromaDB.
    # 0 is the sentinel meaning 'no timestamp' — ChromaDB does not accept None metadata.
    timestamp_unix: Optional[int] = Field(
        default=0,
        description="Unix timestamp (seconds) of the source event, for numeric range queries"
    )

    # Cue classification: "" (primary memory), "topical" (regular cue), "source" (source cue).
    # Distinguishes the two flavors of cue indices in filters and queries.
    cue_type: Optional[str] = Field(
        default="",
        description="Cue classification: '' (primary), 'topical' (regular cue), 'source' (source cue)"
    )

    extra_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata fields (sender, subject, title, source_ref, etc.)"
    )

    def is_cue_index(self) -> bool:
        """True when this entry exists purely as a cue index."""
        return self.linked_memory != ""

    def is_primary_index(self) -> bool:
        """True when this entry is a primary index (not a cue)."""
        return not self.linked_memory

    def get_cue_indices(self) -> List[str]:
        """Return cue indices as a list of trimmed strings."""
        if not self.cue_indices:
            return []
        return [piece.strip() for piece in self.cue_indices.split("||") if piece.strip()]

    def get_predictive_cue_indices(self) -> List[str]:
        """Return predictive (extrinsic) cue indices as a list."""
        if not self.predictive_cue_indices:
            return []
        return [piece.strip() for piece in self.predictive_cue_indices.split("||") if piece.strip()]

    def delete_cue_index(self, cue_index: str):
        """Remove a single cue index from this entry's cue_indices list."""
        remaining = [item for item in self.get_cue_indices() if item != cue_index]
        self.cue_indices = "||".join(remaining)

    def get_linked_memories(self) -> List[str]:
        """Return linked memory indices as a list of strings."""
        if not self.linked_memory:
            return []
        return [piece.strip() for piece in self.linked_memory.split("||") if piece.strip()]

    def get_memory_value(self) -> str:
        """Return the memory's value. Updates embed timestamps in-place, so the field already holds the full history."""
        return self.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Build a MemoryEntry from a plain dictionary."""
        data = dict(data)  # shallow copy so we don't mutate caller's dict
        data["history"] = json.loads(data.get("history", "[]"))
        data["image_urls"] = json.loads(data.get("image_urls", "[]"))
        data["episodic_memory_ids"] = json.loads(data.get("episodic_memory_ids", "[]"))

        # Pull unrecognized keys out into a side bucket so the model accepts the rest cleanly.
        extras: Dict[str, Any] = {}
        for field_name in list(data.keys()):
            if field_name not in _KNOWN_MEMORY_FIELDS:
                extras[field_name] = data.pop(field_name)

        if extras:
            data["extra_metadata"] = extras

        return cls(**data)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Serialize this MemoryEntry to a dict shaped for ChromaDB storage.

        Known fields go to the top level. extra_metadata entries are flattened
        back to the top level so ChromaDB filters (e.g., $contains over sender)
        keep working.

        Returns:
            Dict: Dictionary representation of the memory entry
        """
        metadata = {
            "index": self.index,
            "history": json.dumps(self.history),
            "timestamp": self.timestamp,
            "query": self.query,
            "creation_time": self.creation_time,
            "linked_memory": self.linked_memory,
            "cue_indices": self.cue_indices,
            "image_urls": json.dumps(self.image_urls),
            "memory_type": self.memory_type,
            "episodic_memory_ids": json.dumps(self.episodic_memory_ids),
            # timestamp_unix is always emitted (0 sentinel by default) so ChromaDB numeric filters keep working.
            "timestamp_unix": self.timestamp_unix,
        }
        # Only emit data_type / cue_type / predictive_cue_indices when populated to avoid noise on legacy memories.
        if self.data_type:
            metadata["data_type"] = self.data_type
        if self.cue_type:
            metadata["cue_type"] = self.cue_type
        if self.predictive_cue_indices:
            metadata["predictive_cue_indices"] = self.predictive_cue_indices

        # Flatten extras back into the top-level dict so ChromaDB can filter on them.
        if self.extra_metadata:
            for extra_key, extra_value in self.extra_metadata.items():
                if extra_key not in metadata and extra_value is not None:
                    metadata[extra_key] = extra_value

        return metadata

    def __str__(self) -> str:
        """Compact string representation."""
        chunks = []
        if self.index:
            chunks.append(f"[{self.index}]")
        chunks.append(self.value)
        if self.score > 0:
            chunks.append(f"(score: {self.score:.3f})")
        return " ".join(chunks)

    def __repr__(self) -> str:
        """Verbose representation for debugging."""
        return f"MemoryEntry(index='{self.index}', value='{self.value[:50]}...', score={self.score})"
