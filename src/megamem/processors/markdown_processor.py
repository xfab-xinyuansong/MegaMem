"""
Markdown file processor that turns markdown documents into segments.
"""

import re
from pathlib import Path
from typing import List
from markdownify import markdownify
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class MarkdownProcessor(FileProcessor):
    """Processor for markdown files (.md, .markdown, .mdown, .mdx)."""

    def can_process(self, file_path: Path) -> bool:
        """Return ``True`` for markdown files."""
        return detect_file_type(file_path) == "markdown"

    def process(self, file_path: Path) -> List[Segment]:
        """Read a markdown file and split it into segments.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Ordered list of :class:`Segment` objects.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_markdown_content(content, file_path)

    def _process_markdown_content(self, content: str, file_path: Path) -> List[Segment]:
        """Walk markdown lines and emit one segment per heading section.

        We segment on the original markdown structure first so headings are
        preserved, then strip embedded HTML afterwards.

        Args:
            content: Raw markdown text.
            file_path: Source file (for metadata).

        Returns:
            Ordered list of segments.
        """
        segments: List[Segment] = []
        lines = content.split("\n")
        base_metadata = self._create_base_metadata(file_path, "markdown")

        current_segment_lines: List[str] = []
        current_heading = None
        current_heading_level = None

        # Tracks the parent heading at every level seen so far: {level: text}.
        heading_hierarchy: dict = {}

        heading_re = re.compile(r"^(#{1,6})\s+(.+)")

        for line in lines:
            heading_match = heading_re.match(line.strip())

            if heading_match is None:
                # Non-heading lines simply accumulate into the current segment.
                current_segment_lines.append(line)
                continue

            # Hitting a heading: flush whatever segment we were collecting.
            if current_segment_lines and any(l.strip() for l in current_segment_lines):
                segment_content = "\n".join(current_segment_lines).strip()
                segment_metadata = base_metadata.copy()

                # Heading metadata (uses dummy values when no heading is active).
                segment_metadata = self._add_heading_metadata(
                    segment_metadata,
                    current_heading,
                    current_heading_level,
                    heading_hierarchy,
                )

                # Now safe to strip HTML — structural pass is complete.
                cleaned_segment_content = self._clean_html_tags(segment_content)

                segments.append(
                    Segment(
                        content=cleaned_segment_content,
                        segment_type="section",
                        metadata=segment_metadata,
                    )
                )

            # Begin the next segment under this freshly seen heading.
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            heading_hierarchy[level] = heading_text
            # Drop any deeper hierarchy entries that no longer apply.
            for deeper in [k for k in heading_hierarchy.keys() if k > level]:
                del heading_hierarchy[deeper]

            current_heading = heading_text
            current_heading_level = level
            current_segment_lines = [line]  # keep the heading line itself

        # Flush the trailing segment if one is pending.
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
        """Convert any embedded HTML in ``content`` back into clean markdown."""
        # markdownify handles the bulk of the HTML -> markdown conversion.
        cleaned_content = markdownify(
            content,
            heading_style="ATX",  # use # style headings
            bullets="-",          # use - for bullets
            strip=["script", "style"],  # drop scripts/styles entirely
        )

        # Repair edge cases where markdownify produces glued-together headings.
        cleaned_content = re.sub(r"([.!?])(#{1,6})", r"\1\n\n\2", cleaned_content)
        cleaned_content = re.sub(
            r"(\|)(#{1,6})", r"\1\n\n\2", cleaned_content
        )  # following table rows
        cleaned_content = re.sub(
            r"(\*\*)(#{1,6})", r"\1\n\n\2", cleaned_content
        )  # following bold runs

        # Tidy up whitespace.
        cleaned_content = re.sub(
            r"\n\s*\n\s*\n+", "\n\n", cleaned_content
        )  # collapse runs of blank lines
        cleaned_content = re.sub(
            r"^\s+|\s+$", "", cleaned_content, flags=re.MULTILINE
        )  # strip leading/trailing whitespace per line

        return cleaned_content.strip()

    def _build_heading_path(self, heading_hierarchy: dict, current_level: int) -> str:
        """Render the active heading hierarchy as a ``A > B > C`` path string.

        Args:
            heading_hierarchy: ``{level: heading_text}`` map.
            current_level: Level of the segment whose path we want.

        Returns:
            Hierarchical path such as ``"Chapter 1 > Section A"``.
        """
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
        """Merge heading-related fields into ``metadata``.

        When no heading is active, dummy values are used so the metadata
        schema stays consistent across all segments.

        Args:
            metadata: Base metadata to update.
            current_heading: Heading text (or ``None``).
            current_heading_level: Heading level (or ``None``).
            heading_hierarchy: Active heading hierarchy snapshot.

        Returns:
            The updated metadata dict.
        """
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
            # Fill in placeholder values when no heading is in scope.
            metadata.update(
                {
                    "heading": "",
                    "heading_level": 0,
                    "heading_path": "",
                    "parent_headings": {},
                }
            )

        return metadata
