"""
Registry that wires up file-type processors.
"""

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
    """Holds the set of processors known to the system."""

    def __init__(self):
        """Set up an empty list and register the built-in processors."""
        self._processors: List[BaseProcessor] = []
        self._register_default_processors()

    def _register_default_processors(self):
        """Register every processor that ships with the library."""
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
        """Append ``processor`` to the registry.

        Args:
            processor: A processor instance to make available for dispatch.
        """
        self._processors.append(processor)

    def get_processor(self, file_path: Path) -> Optional[BaseProcessor]:
        """Return the first processor that claims to handle ``file_path``.

        Args:
            file_path: Path of the file we want to process.

        Returns:
            A matching processor, or ``None`` if none accept the file.
        """
        for processor in self._processors:
            if processor.can_process(file_path):
                return processor
        return None

    def process_file(self, file_path: Path) -> List[Segment]:
        """Dispatch ``file_path`` to the appropriate processor and return segments.

        Args:
            file_path: Path of the file to process.

        Returns:
            List of :class:`Segment` objects produced by the processor.

        Raises:
            ValueError: When no processor can handle the file's type.
        """
        processor = self.get_processor(file_path)
        if processor is None:
            raise ValueError(f"No processor found for file type: {file_path.suffix}")

        return processor.process(file_path)

    def get_supported_extensions(self) -> List[str]:
        """Return all file extensions for which a processor is registered.

        Returns:
            Deduplicated list of supported extensions.
        """
        # Probe the registry with a representative file per extension. This is
        # a coarse approach but mirrors the prior behavior.
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


# Module-level processor registry shared across the codebase.
processor_registry = ProcessorRegistry()
