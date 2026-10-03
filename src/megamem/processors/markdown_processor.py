import re
from pathlib import Path
from typing import List
from markdownify import markdownify
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class MarkdownProcessor(FileProcessor):

    def can_process(self, file_path: Path) -> bool:
        return detect_file_type(file_path) == "markdown"

    def process(self, file_path: Path) -> List[Segment]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_markdown_content(content, file_path)

    def _process_markdown_content(self, content: str, file_path: Path) -> List[Segment]:
        segments: List[Segment] = []
        lines = content.split("\n")
        base_metadata = self._create_base_metadata(file_path, "markdown")

        current_segment_lines: List[str] = []
        current_heading = None
        current_heading_level = None

        heading_hierarchy: dict = {}

        heading_re = re.compile(r"^(#{1,6})\s+(.+)")

        for line in lines:
            heading_match = heading_re.match(line.strip())

            if heading_match is None:
                current_segment_lines.append(line)
                continue

            if current_segment_lines and any(l.strip() for l in current_segment_lines):
                segment_content = "\n".join(current_segment_lines).strip()
                segment_metadata = base_metadata.copy()

                segment_metadata = self._add_heading_metadata(
                    segment_metadata,
                    current_heading,
                    current_heading_level,
                    heading_hierarchy,
                )

                cleaned_segment_content = self._clean_html_tags(segment_content)

                segments.append(
                    Segment(
                        content=cleaned_segment_content,
                        segment_type="section",
                        metadata=segment_metadata,
                    )
                )

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            heading_hierarchy[level] = heading_text
            for deeper in [k for k in heading_hierarchy.keys() if k > level]:
                del heading_hierarchy[deeper]

            current_heading = heading_text
            current_heading_level = level
            current_segment_lines = [line]

        if current_segment_lines and any(l.strip() for l in current_segment_lines):
            segment_content = "\n".join(current_segment_lines).strip()
            segment_metadata = base_metadata.copy()

            segment_metadata = self._add_heading_metadata(
                segment_metadata,
                current_heading,
                current_heading_level,
                heading_hierarchy,
            )

            cleaned_segment_content = self._clean_html_tags(segment_content)

            segments.append(
                Segment(
                    content=cleaned_segment_content,
                    segment_type="section",
                    metadata=segment_metadata,
                )
            )

        return segments

    def _clean_html_tags(self, content: str) -> str:
        cleaned_content = markdownify(
            content,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
        )

        cleaned_content = re.sub(r"([.!?])(#{1,6})", r"\1\n\n\2", cleaned_content)
        cleaned_content = re.sub(
            r"(\|)(#{1,6})", r"\1\n\n\2", cleaned_content
        )
        cleaned_content = re.sub(
            r"(\*\*)(#{1,6})", r"\1\n\n\2", cleaned_content
        )

        cleaned_content = re.sub(
            r"\n\s*\n\s*\n+", "\n\n", cleaned_content
        )
        cleaned_content = re.sub(
            r"^\s+|\s+$", "", cleaned_content, flags=re.MULTILINE
        )

        return cleaned_content.strip()

    def _build_heading_path(self, heading_hierarchy: dict, current_level: int) -> str:
        path_parts = [
            heading_hierarchy[level]
            for level in sorted(heading_hierarchy.keys())
            if level <= current_level
        ]

        return " > ".join(path_parts)

    def _add_heading_metadata(
        self,
        metadata: dict,
        current_heading: str,
        current_heading_level: int,
        heading_hierarchy: dict,
    ) -> dict:
        if current_heading:
            heading_path = self._build_heading_path(
                heading_hierarchy, current_heading_level
            )
            metadata.update(
                {
                    "heading": current_heading,
                    "heading_level": current_heading_level,
                    "heading_path": heading_path,
                    "parent_headings": heading_hierarchy.copy(),
                }
            )
        else:
            metadata.update(
                {
                    "heading": "",
                    "heading_level": 0,
                    "heading_path": "",
                    "parent_headings": {},
                }
            )

        return metadata
