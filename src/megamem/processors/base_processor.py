"""
Base file-processor interface used to turn files into segments.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from megamem.core.segment import Segment


# Authoritative mapping of high-level file-type categories to their extensions.
_FILE_TYPE_EXTENSIONS = {
    "markdown": [".md", ".markdown", ".mdown", ".mdx"],
    "word": [".doc", ".docx", ".docm", ".dotx", ".dotm"],
    "excel": [".xls", ".xlsx", ".xlsm", ".xlsb", ".xltx", ".xltm", ".csv"],
    "powerpoint": [".ppt", ".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"],
    "pdf": [".pdf"],
    "text": [".txt", ".text", ".log", ".readme"],
    "richtext": [".rtf"],
    "html": [".html", ".htm", ".xhtml"],
    "xml": [".xml"],
    "json": [".json", ".jsonl"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "config": [".ini", ".cfg", ".conf"],
    "code": [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
    ],
}

# Pre-computed extension -> category lookup for O(1) detection.
_EXTENSION_TO_TYPE = {
    ext: file_type
    for file_type, extensions in _FILE_TYPE_EXTENSIONS.items()
    for ext in extensions
}


def detect_file_type(file_path: Path) -> str:
    """Return the file-type category for ``file_path``.

    Args:
        file_path: File whose extension should be classified.

    Returns:
        One of the keys in :data:`_FILE_TYPE_EXTENSIONS`, or
        ``"unknown"`` if the extension is not registered.
    """
    return _EXTENSION_TO_TYPE.get(file_path.suffix.lower(), "unknown")


def is_supported_file_type(file_path: Path) -> bool:
    """Report whether ``file_path`` has an extension we know how to process.

    Args:
        file_path: File to inspect.

    Returns:
        ``True`` if the extension belongs to a registered category.
    """
    return file_path.suffix.lower() in _EXTENSION_TO_TYPE


def get_supported_extensions() -> Dict[str, List[str]]:
    """Return a defensive copy of the supported-extensions mapping.

    Returns:
        A copy of the ``{category: [extensions, ...]}`` table.
    """
    return _FILE_TYPE_EXTENSIONS.copy()


class BaseProcessor(ABC):
    """Common ancestor for every file processor."""

    ...


class FileProcessor(BaseProcessor):
    """Base class for file processors that convert files to segments."""

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Return ``True`` if this processor can handle ``file_path``.

        Args:
            file_path: Candidate file path.

        Returns:
            ``True`` when the processor is responsible for ``file_path``.
        """
        pass

    @abstractmethod
    def process(self, file_path: Path) -> List[Segment]:
        """Convert ``file_path`` into a list of :class:`Segment` objects.

        Args:
            file_path: File to read and segment.

        Returns:
            Ordered list of segments produced from the file.
        """
        pass

    def _read_file_content(self, file_path: Path) -> str:
        """Read file text with a fallback to ``latin-1`` if UTF-8 fails."""
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except UnicodeDecodeError:
            # UTF-8 failed - retry with a permissive encoding.
            with open(file_path, 'r', encoding='latin-1') as fh:
                return fh.read()

    def _create_base_metadata(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Build the metadata block shared by every segment of a file.

        Args:
            file_path: Path of the source file.
            file_type: Category previously returned by :func:`detect_file_type`.

        Returns:
            Metadata dict with the source path, name, type, and size.
        """
        size = file_path.stat().st_size if file_path.exists() else 0
        return {
            "source_file": str(file_path),
            "file_type": file_type,
            "file_name": file_path.name,
            "file_size": size,
        }
