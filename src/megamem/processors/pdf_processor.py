"""
PDF document processor for .pdf files.
"""

import re
from statistics import median
from pathlib import Path
from typing import Any, Dict, List, Tuple
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class PDFProcessor(FileProcessor):
    """Processor for PDF files (.pdf)."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle PDF files."""
        return detect_file_type(file_path) == "pdf"

    def process(self, file_path: Path) -> List[Segment]:
        """Convert a PDF into per-paragraph and per-table segments.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Ordered list of :class:`Segment` objects.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required to process PDF files. Install with: pip install pdfplumber"
            )

        try:
            segments: List[Segment] = []
            base_metadata = self._create_base_metadata(file_path, "pdf")

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text and text.strip():
                        # Split page text into paragraphs (blank-line delimited).
                        for para_num, paragraph in enumerate(text.split("\n\n"), 1):
                            stripped = paragraph.strip()
                            if not stripped:
                                continue
                            segments.append(
                                Segment(
                                    content=stripped,
                                    segment_type="paragraph",
                                    metadata={
                                        **base_metadata,
                                        "page_number": page_num,
                                        "paragraph_number": para_num,
                                    },
                                )
                            )

                    # Pull each table out as its own segment.
                    for table_num, table in enumerate(page.extract_tables(), 1):
                        if not table:
                            continue
                        # Render the table as tab-separated rows.
                        table_text = "\n".join(
                            "\t".join(str(cell) if cell else "" for cell in row)
                            for row in table
                        )
                        segments.append(
                            Segment(
                                content=table_text,
                                segment_type="table",
                                metadata={
                                    **base_metadata,
                                    "page_number": page_num,
                                    "table_number": table_num,
                                    "row_count": len(table),
                                    "column_count": len(table[0]) if table else 0,
                                },
                            )
                        )

            return segments

        except Exception as exc:
            raise ValueError(f"Failed to process PDF file: {exc}")

    def extract_header_structure(self, file_path: Path) -> List[Segment]:
        """Return the PDF's heading hierarchy as ordered header segments.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Header segments in reading order.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            records = self._extract_pymupdf_line_records(file_path)
        except ImportError:
            raise ImportError(
                "PyMuPDF is required to extract PDF header structure. Install with: pip install pymupdf"
            )

        if not records:
            return []

        body_font_size = self._estimate_body_font_size(records)
        base_metadata = self._create_base_metadata(file_path, "pdf")
        seen: set[Tuple[int, str]] = set()
        header_candidates: List[Tuple[int, float, int, str]] = []
        in_references = False

        numbered_re = re.compile(r"^\d+(?:\.\d+){0,3}\s+")
        subsection_re = re.compile(r"^\d+\.\d+(?:\.\d+){0,2}\s+")
        top_level_re = re.compile(r"^\d+\s+")

        for record in records:
            page_number = int(record["page"])
            line_text = str(record["text"])
            font_size = float(record["size"])
            is_emphasis = bool(record["emphasis"])
            y_pos = float(record["y"])

            candidates = [self._clean_header_text(line_text)]
            candidates.extend(self._extract_inline_numbered_candidates(line_text))

            for candidate in candidates:
                if not self._looks_like_header(candidate):
                    continue

                is_numbered = bool(numbered_re.match(candidate))
                is_subsection = bool(subsection_re.match(candidate))
                is_top_level_numbered = bool(top_level_re.match(candidate))
                has_visual_header_signal = is_emphasis or font_size >= body_font_size + 0.6

                # Subsections must have a visual emphasis cue.
                if is_subsection and not has_visual_header_signal:
                    continue

                # Top-level numbered headings need a stronger visual cue.
                if is_top_level_numbered and not (is_emphasis or font_size >= body_font_size + 1.5):
                    continue

                # Anything not numbered must at least have a visual cue.
                if not is_numbered and not has_visual_header_signal:
                    continue

                # Once we enter the references section, only appendices count.
                if in_references and not candidate.lower().startswith("appendix"):
                    continue

                dedupe_key = (page_number, candidate.lower())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                level = self._header_level(candidate)
                header_candidates.append((page_number, y_pos, level, candidate))

                if candidate.lower().strip(" :") == "references":
                    in_references = True

        header_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        return [
            Segment(
                content=header_text,
                segment_type="header",
                metadata={
                    **base_metadata,
                    "page_number": page_number,
                    "heading_level": level,
                },
            )
            for page_number, _y_pos, level, header_text in header_candidates
        ]

    def _normalize_header_text(self, line: str) -> str:
        """Compact whitespace and re-space leading numbering on a header line."""
        text = re.sub(r"\s+", " ", line.strip())
        # Insert a space between the section number and the title in three
        # progressively more permissive forms.
        text = re.sub(r"^(\d+(?:\.\d+)*)(?=[A-Za-z])", r"\1 ", text)
        text = re.sub(r"^(\d+(?:\.\d+)*)\.\s+(?=[A-Za-z])", r"\1 ", text)
        text = re.sub(r"^(\d+(?:\.\d+)*)\.(?=[A-Za-z])", r"\1 ", text)
        return text

    def _is_probable_non_header(self, text: str) -> bool:
        """Heuristic blacklist for lines that almost certainly aren't headers."""
        text_len = len(text)
        if text_len < 4 or text_len > 95:
            return True

        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        spaces = text.count(" ")
        denom = max(1, text_len)

        if letters < 3:
            return True
        if letters / denom < 0.45:
            return True
        if digits / denom > 0.30:
            return True
        if symbols / denom > 0.22:
            return True

        lower = text.lower()
        if "http://" in lower or "https://" in lower or "www." in lower:
            return True

        if any(ch in text for ch in ["±", "=", "⊤", "≤", "≥", "Ω", "µ", "σ", "∈", "←", "⇒"]):
            return True

        if "," in text and text.count(",") >= 4:
            return True

        if digits >= 5 and spaces >= 3:
            return True

        if spaces == 0 and text_len > 28:
            return True

        if text.endswith(".") and not re.match(r"^\d+(?:\.\d+)*\s+", text):
            return True

        return False

    def _looks_like_header(self, line: str) -> bool:
        """Decide whether ``line`` plausibly represents a section header."""
        text = self._normalize_header_text(line)
        lower = text.lower().strip(" :")

        # Numbered subsections like "2.3.1 Foo bar".
        subsection_match = re.match(r"^(\d+(?:\.\d+){1,3})\s+[A-Z]", text)
        if subsection_match and self._is_valid_section_number(subsection_match.group(1)):
            subsection_title = re.sub(r"^\d+(?:\.\d+){1,3}\s+", "", text).strip()
            sub_tokens = subsection_title.split()
            first_word = sub_tokens[0].lower() if sub_tokens else ""
            if first_word in {"let", "if", "when", "where", "while", "then", "for"}:
                return False
            if re.search(r"[=≤≥∈δµσ⊤Ω←⇒]", subsection_title):
                return False
            if subsection_title.endswith("."):
                return False
            if not (1 <= len(sub_tokens) <= 8):
                return False
            return True

        # Top-level numbered sections like "1 Introduction".
        section_text = self._prettify_header_text(text)
        section_match = re.match(r"^(\d+)\s+(.+)$", section_text)
        if section_match and self._is_valid_section_number(section_match.group(1)):
            section_title = section_match.group(2).strip()
            sec_tokens = section_title.split()
            if section_title and section_title[0].isupper() and 1 <= len(sec_tokens) <= 10:
                if not re.search(r"[=≤≥∈δµσ⊤Ω←⇒]", section_title):
                    if not section_title.endswith("."):
                        return True

        if self._is_probable_non_header(text):
            return False

        if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z][A-Za-z0-9\-,:() ]{2,80}$", text):
            return True

        if re.match(r"^[IVXLC]+\.?\s+[A-Z][A-Za-z0-9\-,:() ]{2,80}$", text):
            return True

        if re.match(r"^(appendix\s+[A-Z0-9]+|appendix)\b", lower):
            return True

        return False

    def _header_level(self, line: str) -> int:
        """Return the heading depth implied by ``line`` (defaults to 1)."""
        text = self._normalize_header_text(line)
        match = re.match(r"^(\d+(?:\.\d+)*)\s+", text)
        if match:
            return match.group(1).count(".") + 1
        # Roman-numeral or unparseable forms collapse to top level.
        return 1

    def _clean_header_text(self, line: str) -> str:
        """Trim ``line`` after the section prefix down to a clean header title."""
        text = self._normalize_header_text(line)

        numbered_match = re.match(r"^(\d+(?:\.\d+){0,3}|[IVXLC]+\.?)(\s+)(.+)$", text)
        if not numbered_match:
            return self._prettify_header_text(text)

        prefix = numbered_match.group(1)
        remainder = numbered_match.group(3).strip()

        cleaned_tokens: List[str] = []
        # Walk word-by-word and stop on common sentence-internal markers.
        for token in remainder.split():
            if "," in token or ";" in token or "?" in token:
                break
            if token.endswith(":"):
                cleaned_tokens.append(token.rstrip(":"))
                break
            cleaned_tokens.append(token)
            if len(cleaned_tokens) >= 8:
                break

        if not cleaned_tokens:
            return self._prettify_header_text(text)

        return self._prettify_header_text(f"{prefix} {' '.join(cleaned_tokens)}".strip())

    def _prettify_header_text(self, text: str) -> str:
        """Normalize spacing inside a header text snippet."""
        pretty = re.sub(r"\s+", " ", text.strip())
        # Insert spaces between camel-case / alphanumeric boundaries.
        pretty = re.sub(r"([a-z])([A-Z])", r"\1 \2", pretty)
        pretty = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", pretty)
        pretty = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", pretty)
        return pretty

    def _is_valid_section_number(self, section_number: str) -> bool:
        """Return ``True`` when ``section_number`` looks like a real section index."""
        parts = section_number.split(".")
        if not parts:
            return False

        try:
            values = [int(part) for part in parts]
        except ValueError:
            return False

        # Each component must lie within a plausible 1..30 range.
        return all(1 <= v <= 30 for v in values)

    def _extract_inline_numbered_candidates(self, line: str) -> List[str]:
        """Pull header-like fragments that begin partway through ``line``."""
        candidates: List[str] = []
        normalized = self._normalize_header_text(line)

        for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+){1,3})\.?\s*(?=[A-Za-z])", normalized):
            if not self._is_valid_section_number(match.group(1)):
                continue
            snippet = normalized[match.start():].strip()
            if snippet:
                candidates.append(self._clean_header_text(snippet))

        return candidates

    def _extract_pymupdf_line_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """Walk the PDF via PyMuPDF and emit one record per text line."""
        import fitz

        records: List[Dict[str, Any]] = []
        document = fitz.open(file_path)
        try:
            for page_index, page in enumerate(document, start=1):
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    # ``type`` 0 is a text block; skip everything else.
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        joined = "".join(span.get("text", "") for span in spans).strip()
                        normalized = self._normalize_header_text(joined)
                        if not normalized:
                            continue

                        font_size = max(float(span.get("size", 0.0)) for span in spans)
                        is_emphasis = any(
                            marker in str(span.get("font", "")).lower()
                            for span in spans
                            for marker in ("bold", "medi", "black")
                        )
                        bbox = line.get("bbox", [0.0, 0.0, 0.0, 0.0])

                        records.append(
                            {
                                "page": page_index,
                                "text": normalized,
                                "size": font_size,
                                "emphasis": is_emphasis,
                                "y": float(bbox[1]),
                            }
                        )
        finally:
            document.close()

        return records

    def _estimate_body_font_size(self, records: List[Dict[str, Any]]) -> float:
        """Pick a baseline font size representative of body text."""
        # Long-ish lines that contain lowercase letters are probably body text.
        sample_sizes: List[float] = [
            float(rec["size"])
            for rec in records
            if len(str(rec["text"])) >= 45 and any(ch.islower() for ch in str(rec["text"]))
        ]

        if not sample_sizes:
            sample_sizes = [float(rec["size"]) for rec in records]

        return median(sample_sizes) if sample_sizes else 10.0
