from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Segment:
    content: str
    segment_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        snippet = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Segment({self.segment_type}): {snippet}"

    def __repr__(self) -> str:
        return f"Segment(type={self.segment_type}, content_len={len(self.content)})"
