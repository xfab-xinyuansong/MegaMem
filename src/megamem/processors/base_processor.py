from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from megamem.core.segment import Segment


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

_EXTENSION_TO_TYPE = {
    ext: file_type
    for file_type, extensions in _FILE_TYPE_EXTENSIONS.items()
    for ext in extensions
}


def detect_file_type(file_path: Path) -> str:
    return _EXTENSION_TO_TYPE.get(file_path.suffix.lower(), "unknown")


def is_supported_file_type(file_path: Path) -> bool:
    return file_path.suffix.lower() in _EXTENSION_TO_TYPE


def get_supported_extensions() -> Dict[str, List[str]]:
    return _FILE_TYPE_EXTENSIONS.copy()


class BaseProcessor(ABC):

    ...


class FileProcessor(BaseProcessor):

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def process(self, file_path: Path) -> List[Segment]:
        pass

    def _read_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as fh:
                return fh.read()

    def _create_base_metadata(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        size = file_path.stat().st_size if file_path.exists() else 0
        return {
            "source_file": str(file_path),
            "file_type": file_type,
            "file_name": file_path.name,
            "file_size": size,
        }
