import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import tiktoken


def index_to_id(key: str) -> str:
    # Deterministic hash → record id mapping.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def count_tokens(content: str) -> int:
    """Count tokens with the package's default ``cl100k_base`` encoding."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(content))


def normalize_content(
    content: Union[str, List[str], List[Dict[str, Any]]],
    multimodal_support: bool = True,
):
    """Normalise heterogeneous context input into a uniform shape.

    Handles plain text, lists of strings and conversation/multimodal
    message lists, and produces a dict suitable for downstream consumers.

    Args:
        content: input in any of the supported shapes:
            - ``str``: a single text blob.
            - ``List[str]``: several strings to concatenate.
            - ``List[Dict]``: chat-style messages, possibly with
              embedded multimodal content parts.
        multimodal_support: when ``True`` and image parts are present,
            the returned dict includes an ``image`` key.

    Returns:
        Text-only payload::

            {"text": "User: Hello\\nAssistant: Hi there!"}

        Multimodal payload (when supported and present)::

            {
                "text": "...",
                "image": [{"type": "image_url", ...}],
            }

    Raises:
        ValueError: when the content shape isn't supported.
    """
    text_parts: List[str] = []
    image_parts: List[Dict[str, Any]] = []

    if isinstance(content, str):
        text_parts.append(content.strip())
    elif isinstance(content, list):
        if all(isinstance(item, str) for item in content):
            text_parts.extend(content)
        elif all(isinstance(item, dict) for item in content):
            if any("role" in item and "content" in item for item in content):
                for turn_idx, msg in enumerate(content, start=1):
                    if isinstance(msg, dict) and "content" in msg:
                        msg_content = msg["content"]

                        if isinstance(msg_content, list):
                            chunks: List[str] = []
                            for part in msg_content:
                                if isinstance(part, dict):
                                    ptype = part.get("type")
                                    if ptype == "text":
                                        chunks.append(part.get("text", ""))
                                    elif ptype == "image_url":
                                        image_parts.append(part)
                            text_parts.append(f"[Turn {turn_idx}] {' '.join(chunks)}")
                        elif isinstance(msg_content, str):
                            text_parts.append(f"[Turn {turn_idx}] {msg_content}")
            else:
                text_parts.extend([str(item) for item in content])
        else:
            raise ValueError(
                "Context list must contain either all strings or all dictionaries"
            )
    else:
        raise ValueError(
            "Context must be a string, list of strings, or list of dictionaries"
        )

    segment_messages = None
    if isinstance(content, list) and all(isinstance(item, dict) for item in content):
        if any("role" in item and "content" in item for item in content):
            segment_messages = content

    if image_parts and multimodal_support:
        return {
            "text": "\n".join(text_parts),
            "image": image_parts,
            "segment_messages": segment_messages,
        }
    return {
        "text": "\n".join(text_parts),
        "segment_messages": segment_messages,
    }


def context_to_str(
    context: Union[str, List[str], List[Dict[str, str]]],
):
    """Backward-compatible string-only adapter around ``normalize_content``.

    Image parts are dropped. Prefer ``normalize_content`` directly when
    writing new code.
    """
    return normalize_content(context, multimodal_support=False)["text"]


def add_and_condition(where: Optional[dict], new_condition: dict) -> dict:
    if where is None:
        return new_condition
    if "$and" in where:
        where["$and"].append(new_condition)
        return where
    return {"$and": [where, new_condition]}


# def get_current_timestamp() -> str:
#     """
#     Get the current timestamp as a formatted string.
#
#     Returns:
#         Current timestamp in ISO format (YYYY-MM-DD HH:MM:SS)
#     """
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_timestamp() -> str:
    """Current local time formatted as ``YYYY-MM-DD HH:MM:SS``.

    Returns:
        Current timestamp in ISO 8601 format with UTC timezone (YYYY-MM-DDTHH:MM:SS.mmmZ)
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_user_id_from_where(where: Optional[dict]) -> Optional[str]:
    """Pull a ``user_id`` value out of a ChromaDB-style ``where`` clause.

    Args:
        where: filter dict, possibly nested under ``$and``.

    Returns:
        The first ``user_id`` value found, or ``None``.
    """
    if where is None:
        return None
    if "user_id" in where:
        return where["user_id"]
    if "$and" in where:
        for cond in where["$and"]:
            if "user_id" in cond:
                return cond["user_id"]
    return None


def merge_metadata(
    segment_metadata: Optional[Dict], user_metadata: Optional[Dict]
) -> Dict:
    """Combine segment-derived metadata with user-supplied metadata.

    User metadata takes precedence on key collisions.

    Args:
        segment_metadata: metadata extracted from the segment.
        user_metadata: metadata provided by the caller.

    Returns:
        Merged metadata dictionary.
    """
    merged: Dict = {}
    if segment_metadata:
        merged.update(segment_metadata)
    if user_metadata:
        merged.update(user_metadata)
    return merged


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extension_to_type(extension: str) -> str:
    """Translate a file extension into a memory-builder file-type tag.

    Args:
        extension: extension string (with or without leading dot).

    Returns:
        File-type tag understood by the memory-builder selector. Falls
        back to ``"text"`` for unknown extensions.
    """
    extension = extension.lower().strip(".")

    ext_map = {
        # Text files
        "txt": "text",
        "md": "markdown",
        "markdown": "markdown",
        # Document files
        "doc": "word",
        "docx": "word",
        "pdf": "pdf",
        "rtf": "text",
        # Spreadsheet files
        "xls": "excel",
        "xlsx": "excel",
        "csv": "table",
        # Presentation files
        "ppt": "powerpoint",
        "pptx": "powerpoint",
        # Web files
        "html": "html",
        "htm": "html",
        "xml": "xml",
        # Data files
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        # Code files (treat as text)
        "py": "text",
        "js": "text",
        "ts": "text",
        "java": "text",
        "cpp": "text",
        "c": "text",
        "h": "text",
        "css": "text",
        "sql": "text",
    }

    return ext_map.get(extension, "text")
