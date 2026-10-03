from pathlib import Path
from typing import List, Dict, Type, Optional
from megamem.core.segment import Segment
from megamem.processors.base_processor import BaseProcessor
from megamem.processors.excel_processor import ExcelProcessor
from megamem.processors.markdown_processor import MarkdownProcessor
from megamem.processors.pdf_processor import PDFProcessor
from megamem.processors.powerpoint_processor import PowerPointProcessor
from megamem.processors.text_processor import TextProcessor
from megamem.processors.word_processor import WordProcessor


class ProcessorRegistry:

    def __init__(self):
        self._processors: List[BaseProcessor] = []
        self._register_default_processors()

    def _register_default_processors(self):
        for proc in (
            MarkdownProcessor(),
            TextProcessor(),
            WordProcessor(),
            ExcelProcessor(),
            PowerPointProcessor(),
            PDFProcessor(),
        ):
            self.register(proc)

    def register(self, processor: BaseProcessor):
        self._processors.append(processor)

    def get_processor(self, file_path: Path) -> Optional[BaseProcessor]:
        for processor in self._processors:
            if processor.can_process(file_path):
                return processor
        return None

    def process_file(self, file_path: Path) -> List[Segment]:
        processor = self.get_processor(file_path)
        if processor is None:
            raise ValueError(f"No processor found for file type: {file_path.suffix}")

        return processor.process(file_path)

    def get_supported_extensions(self) -> List[str]:
        test_files = [
            Path("test.md"), Path("test.markdown"),
            Path("test.txt"), Path("test.docx"),
            Path("test.pdf"), Path("test.xlsx"),
        ]

        extensions = [
            tf.suffix
            for tf in test_files
            if self.get_processor(tf) is not None
        ]

        return list(set(extensions))


processor_registry = ProcessorRegistry()
