from pathlib import Path
from typing import List, Dict, Any, Optional
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class WordProcessor(FileProcessor):

    def __init__(self, max_segment_size: int = 1000):
        self.max_segment_size = max_segment_size

    def can_process(self, file_path: Path) -> bool:
        return detect_file_type(file_path) == "word"

    def process(self, file_path: Path) -> List[Segment]:
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
                try:
                    level = int(paragraph.style.name.split()[-1])
                except (ValueError, IndexError):
                    level = 1

                if current_content_parts and len(current_content_parts) > 1:
                    self._save_segment(
                        segments,
                        current_content_parts,
                        current_heading,
                        current_heading_level,
                        heading_hierarchy.copy(),
                        base_metadata,
                    )

                heading_hierarchy[level] = text
                for deeper in [k for k in heading_hierarchy.keys() if k > level]:
                    del heading_hierarchy[deeper]

                current_heading = text
                current_heading_level = level
                current_content_parts = [text]

            else:
                current_size = sum(len(part) for part in current_content_parts)
                would_overflow = (
                    current_content_parts
                    and len(current_content_parts) > 1
                    and current_size + len(text) + 1 > self.max_segment_size
                )

                if would_overflow:
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
        if heading is None:
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
        path_parts = [
            heading_hierarchy[level]
            for level in sorted(heading_hierarchy.keys())
            if level <= current_level
        ]

        return " > ".join(path_parts)
