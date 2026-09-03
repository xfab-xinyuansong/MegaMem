"""
Word document processor for .doc and .docx files.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class WordProcessor(FileProcessor):
    """Processor for Microsoft Word documents (.doc, .docx)."""

    def __init__(self, max_segment_size: int = 1000):
        """
        Initialize the Word processor.

        Args:
            max_segment_size: Maximum size of each segment in characters
                (default: 1000). Combining of sibling sections is currently
                disabled and was previously controlled here.
        """
        self.max_segment_size = max_segment_size

    def can_process(self, file_path: Path) -> bool:
        """Return ``True`` for Word documents."""
        return detect_file_type(file_path) == "word"

    def process(self, file_path: Path) -> List[Segment]:
        """Convert a Word document into a list of segments.

        Each segment groups a heading with its following paragraphs and is
        cut whenever the running size exceeds ``max_segment_size`` (paragraph
        boundaries are respected).

        Args:
            file_path: Path to the Word file.

        Returns:
            Ordered list of :class:`Segment` objects.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required to process Word documents. Install with: pip install python-docx"
            )

        try:
            doc = Document(file_path)
            base_metadata = self._create_base_metadata(file_path, "word")

            return self._combine_paragraphs_into_segments(doc, base_metadata)

        except Exception as exc:
            raise ValueError(f"Failed to process Word document: {exc}")

    def _combine_paragraphs_into_segments(
        self, doc, base_metadata: Dict[str, Any]
    ) -> List[Segment]:
        """Pack headings + their paragraphs into bounded-size segments.

        Mirrors the markdown processor's "heading + content" approach so that
        a heading travels with the content it introduces.

        Args:
            doc: ``Document`` object from python-docx.
            base_metadata: Metadata applied to every produced segment.

        Returns:
            Ordered list of :class:`Segment` objects.
        """
        segments: List[Segment] = []
        current_content_parts: List[str] = []
        current_heading: Optional[str] = None
        current_heading_level: Optional[int] = None
        heading_hierarchy: Dict[int, str] = {}

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            is_heading = paragraph.style.name.startswith("Heading")

            if is_heading:
                # Pull the numeric level out of the style name (default to 1).
                try:
                    level = int(paragraph.style.name.split()[-1])
                except (ValueError, IndexError):
                    level = 1

                # Persist the previous segment unless it only had its heading.
                if current_content_parts and len(current_content_parts) > 1:
                    self._save_segment(
                        segments,
                        current_content_parts,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy.copy(),
                        base_metadata,
                    )

                # Refresh the heading hierarchy.
                heading_hierarchy[level] = text
                # Drop any stale deeper levels.
                for deeper in [k for k in heading_hierarchy.keys() if k > level]:
                    del heading_hierarchy[deeper]

                # Open a fresh segment with this heading.
                current_heading = text
                current_heading_level = level
                current_content_parts = [text]

            else:
                # Decide whether the new paragraph would overflow the segment.
                current_size = sum(len(part) for part in current_content_parts)
                would_overflow = (
                    current_content_parts
                    and len(current_content_parts) > 1
                    and current_size + len(text) + 1 > self.max_segment_size
                )

                if would_overflow:
                    # Flush the existing segment and start fresh while keeping
                    # the heading context implicitly via heading_hierarchy.
                    self._save_segment(
                        segments,
                        current_content_parts,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy.copy(),
                        base_metadata,
                    )
                    current_content_parts = [text]
                else:
                    current_content_parts.append(text)

        # Persist the trailing segment if it carries actual content.
        if current_content_parts and len(current_content_parts) > 1:
            self._save_segment(
                segments,
                current_content_parts,
                current_heading,
                current_heading_level,
                heading_hierarchy.copy(),
                base_metadata,
            )

        return segments

    def _save_segment(
        self,
        segments: List[Segment],
        content_parts: List[str],
        heading: Optional[str],
        heading_level: Optional[int],
        heading_hierarchy: Dict[int, str],
        base_metadata: Dict[str, Any],
    ) -> None:
        """Append a single segment built from the supplied parts/metadata."""
        if not content_parts:
            return

        segment_content = "\n\n".join(content_parts).strip()

        metadata = base_metadata.copy()
        metadata.update(self._build_heading_metadata(heading, heading_level, heading_hierarchy))

        segments.append(
            Segment(
                content=segment_content,
                segment_type="section",
                metadata=metadata,
            )
        )

    def _build_heading_metadata(
        self,
        heading: Optional[str],
        heading_level: Optional[int],
        heading_hierarchy: Dict[int, str],
    ) -> Dict[str, Any]:
        """Render the heading-related metadata for a segment.

        Args:
            heading: Active heading text (or ``None``).
            heading_level: Active heading level (or ``None``).
            heading_hierarchy: Snapshot of the active heading stack.

        Returns:
            Dict of heading-related metadata fields.
        """
        if heading is None:
            # Placeholders so the schema stays uniform across all segments.
            return {
                "heading": "",
                "heading_level": 0,
                "heading_path": "",
                "parent_headings": {},
            }

        heading_path = self._build_heading_path(heading_hierarchy, heading_level)
        return {
            "heading": heading,
            "heading_level": heading_level,
            "heading_path": heading_path,
            "parent_headings": heading_hierarchy.copy(),
        }

    def _build_heading_path(
        self, heading_hierarchy: Dict[int, str], current_level: int
    ) -> str:
        """Render the heading hierarchy as a ``A > B > C`` path string.

        Args:
            heading_hierarchy: ``{level: heading_text}`` map.
            current_level: Level of the segment whose path we want.

        Returns:
            Path such as ``"Chapter 1 > Section A > Subsection 3"``.
        """
        path_parts = [
            heading_hierarchy[level]
            for level in sorted(heading_hierarchy.keys())
            if level <= current_level
        ]

        return " > ".join(path_parts)
