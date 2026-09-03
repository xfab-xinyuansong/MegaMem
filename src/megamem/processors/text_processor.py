"""
Plain-text file processor that converts text files into segments.
"""

from pathlib import Path
from typing import List
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class TextProcessor(FileProcessor):
    """Processor for plain text files (.txt, .text, .log, etc.)."""

    def can_process(self, file_path: Path) -> bool:
        """Return ``True`` for files classified as ``"text"``."""
        return detect_file_type(file_path) == 'text'

    def process(self, file_path: Path) -> List[Segment]:
        """Read ``file_path`` and convert it into paragraph segments.

        Args:
            file_path: Path to the text file.

        Returns:
            Ordered list of :class:`Segment` objects.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_text_content(content, file_path)

    def _process_text_content(self, content: str, file_path: Path) -> List[Segment]:
        """Split ``content`` on blank lines and emit one segment per paragraph."""
        paragraphs = content.split('\n\n')
        segments: List[Segment] = []
        base_metadata = self._create_base_metadata(file_path, "text")

        for pos, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            segments.append(
                Segment(
                    content=paragraph.strip(),
                    segment_type="paragraph",
                    metadata={
                        **base_metadata,
                        "paragraph_number": pos + 1,
                    },
                )
            )

        # Fallback: no blank lines found, emit the whole file as one segment.
        if not segments and content.strip():
            segments.append(
                Segment(
                    content=content.strip(),
                    segment_type="text",
                    metadata=base_metadata,
                )
            )

        return segments
