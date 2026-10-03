import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class MemoryMetadata(BaseModel):
    pass


_KNOWN_MEMORY_FIELDS = {
    "value", "index", "history", "memory_type", "episodic_memory_ids",
    "score", "timestamp", "query", "creation_time", "linked_memory",
    "cue_indices", "predictive_cue_indices", "image_urls", "data_type",
    "timestamp_unix", "extra_metadata", "cue_type",
}


class MemoryEntry(BaseModel):

    value: str = Field(
        description="The main memory content/text"
    )

    index: Optional[str] = Field(
        default="",
        description="Memory index/key for retrieval and identification"
    )

    history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Historical versions of this memory entry"
    )

    memory_type: Optional[str] = Field(
        default="",
        description="Type of memory (e.g., 'factual', 'procedural', 'episodic')"
    )

    episodic_memory_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="List of episodic memory IDs that provide context for this factual memory"
    )

    score: Optional[float] = Field(
        default=0.0,
        description="Relevance/similarity score"
    )

    timestamp: Optional[str] = Field(
        default="",
        description="Timestamp when the memory event occurred"
    )

    query: Optional[str] = Field(
        default="",
        description="The original query that retrieved this memory"
    )

    creation_time: Optional[str] = Field(
        default="",
        description="Timestamp when the memory was originally created"
    )

    linked_memory: Optional[str] = Field(
        default="",
        description="Reference to linked memory entries"
    )

    cue_indices: Optional[str] = Field(
        default="", description="Indices linked to this memory for enhanced retrieval"
    )

    predictive_cue_indices: Optional[str] = Field(
        default="",
        description="Extrinsic cue indices for bridging semantically distant queries",
    )

    image_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="List of image URLs associated with this memory"
    )

    data_type: Optional[str] = Field(
        default="",
        description="Source type category: 'mail', 'doc', or '' (unspecified)"
    )

    timestamp_unix: Optional[int] = Field(
        default=0,
        description="Unix timestamp (seconds) of the source event, for numeric range queries"
    )

    cue_type: Optional[str] = Field(
        default="",
        description="Cue classification: '' (primary), 'topical' (regular cue), 'source' (source cue)"
    )

    extra_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata fields (sender, subject, title, source_ref, etc.)"
    )

    def is_cue_index(self) -> bool:
        return self.linked_memory != ""

    def is_primary_index(self) -> bool:
        return not self.linked_memory

    def get_cue_indices(self) -> List[str]:
        if not self.cue_indices:
            return []
        return [piece.strip() for piece in self.cue_indices.split("||") if piece.strip()]

    def get_predictive_cue_indices(self) -> List[str]:
        if not self.predictive_cue_indices:
            return []
        return [piece.strip() for piece in self.predictive_cue_indices.split("||") if piece.strip()]

    def delete_cue_index(self, cue_index: str):
        remaining = [item for item in self.get_cue_indices() if item != cue_index]
        self.cue_indices = "||".join(remaining)

    def get_linked_memories(self) -> List[str]:
        if not self.linked_memory:
            return []
        return [piece.strip() for piece in self.linked_memory.split("||") if piece.strip()]

    def get_memory_value(self) -> str:
        return self.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        data = dict(data)
        data["history"] = json.loads(data.get("history", "[]"))
        data["image_urls"] = json.loads(data.get("image_urls", "[]"))
        data["episodic_memory_ids"] = json.loads(data.get("episodic_memory_ids", "[]"))

        extras: Dict[str, Any] = {}
        for field_name in list(data.keys()):
            if field_name not in _KNOWN_MEMORY_FIELDS:
                extras[field_name] = data.pop(field_name)

        if extras:
            data["extra_metadata"] = extras

        return cls(**data)

    def get_metadata(self) -> Dict[str, Any]:
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
            "timestamp_unix": self.timestamp_unix,
        }
        if self.data_type:
            metadata["data_type"] = self.data_type
        if self.cue_type:
            metadata["cue_type"] = self.cue_type
        if self.predictive_cue_indices:
            metadata["predictive_cue_indices"] = self.predictive_cue_indices

        if self.extra_metadata:
            for extra_key, extra_value in self.extra_metadata.items():
                if extra_key not in metadata and extra_value is not None:
                    metadata[extra_key] = extra_value

        return metadata

    def __str__(self) -> str:
        chunks = []
        if self.index:
            chunks.append(f"[{self.index}]")
        chunks.append(self.value)
        if self.score > 0:
            chunks.append(f"(score: {self.score:.3f})")
        return " ".join(chunks)

    def __repr__(self) -> str:
        return f"MemoryEntry(index='{self.index}', value='{self.value[:50]}...', score={self.score})"
