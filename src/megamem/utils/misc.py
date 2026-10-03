import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import tiktoken


def index_to_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def count_tokens(content: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(content))


def normalize_content(
    content: Union[str, List[str], List[Dict[str, Any]]],
    multimodal_support: bool = True,
):
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
    return normalize_content(context, multimodal_support=False)["text"]


def add_and_condition(where: Optional[dict], new_condition: dict) -> dict:
    if where is None:
        return new_condition
    if "$and" in where:
        where["$and"].append(new_condition)
        return where
    return {"$and": [where, new_condition]}


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_user_id_from_where(where: Optional[dict]) -> Optional[str]:
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
    merged: Dict = {}
    if segment_metadata:
        merged.update(segment_metadata)
    if user_metadata:
        merged.update(user_metadata)
    return merged


def extension_to_type(extension: str) -> str:
    extension = extension.lower().strip(".")

    ext_map = {
        "txt": "text",
        "md": "markdown",
        "markdown": "markdown",
        "doc": "word",
        "docx": "word",
        "pdf": "pdf",
        "rtf": "text",
        "xls": "excel",
        "xlsx": "excel",
        "csv": "table",
        "ppt": "powerpoint",
        "pptx": "powerpoint",
        "html": "html",
        "htm": "html",
        "xml": "xml",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
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
