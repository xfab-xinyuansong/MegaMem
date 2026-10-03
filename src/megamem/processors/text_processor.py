from pathlib import Path
from typing import List
from megamem.core.segment import Segment
from megamem.processors.base_processor import FileProcessor, detect_file_type


class TextProcessor(FileProcessor):

    def can_process(self, file_path: Path) -> bool:
        return detect_file_type(file_path) == 'text'

    def process(self, file_path: Path) -> List[Segment]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._read_file_content(file_path)
        return self._process_text_content(content, file_path)

    def _process_text_content(self, content: str, file_path: Path) -> List[Segment]:
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

        if not segments and content.strip():
            segments.append(
                Segment(
                    content=content.strip(),
                    segment_type="text",
                    metadata=base_metadata,
                )
            )

        return segments
