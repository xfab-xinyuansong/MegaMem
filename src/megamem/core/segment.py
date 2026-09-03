"""
Lightweight Segment dataclass that pairs a chunk of text with its kind and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Segment:
    """
    Minimal container for a piece of content together with its type and metadata.
    """
    content: str
    segment_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render the segment as a short, human-readable string."""
        snippet = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Segment({self.segment_type}): {snippet}"

    def __repr__(self) -> str:
        """Verbose form for debugging."""
        return f"Segment(type={self.segment_type}, content_len={len(self.content)})"
