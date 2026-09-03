"""Load the public EnterpriseRAG extension from Hugging Face.

The dependency on :mod:`datasets` is optional and is imported only when one of
the loaders is called. Streaming is enabled by default because the document
configuration is intended for corpus-scale experiments.
"""

from __future__ import annotations

from typing import Any

DATASET_ID = "xsong69/enterpriseRAG-extension"
DOCUMENT_CONFIG = "documents"
QUESTION_CONFIG = "questions"


def _load_dataset_function() -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        raise ImportError(
            'Hugging Face dataset loading requires the optional dependency. '
            'Install it with: pip install -e ".[huggingface]"'
        ) from exc
    return load_dataset


def _load_config(
    config: str,
    *,
    split: str,
    streaming: bool,
    revision: str | None,
    token: str | bool | None,
    **kwargs: Any,
) -> Any:
    options: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        **kwargs,
    }
    if revision is not None:
        options["revision"] = revision
    if token is not None:
        options["token"] = token
    return _load_dataset_function()(DATASET_ID, config, **options)


def load_enterprise_rag_documents(
    *,
    split: str = "test",
    streaming: bool = True,
    revision: str | None = None,
    token: str | bool | None = None,
    **kwargs: Any,
) -> Any:
    """Load source documents with ``doc_id``, type, title, and content fields."""
    return _load_config(
        DOCUMENT_CONFIG,
        split=split,
        streaming=streaming,
        revision=revision,
        token=token,
        **kwargs,
    )


def load_enterprise_rag_questions(
    *,
    split: str = "test",
    streaming: bool = True,
    revision: str | None = None,
    token: str | bool | None = None,
    **kwargs: Any,
) -> Any:
    """Load benchmark questions, references, expected sources, and answer facts."""
    return _load_config(
        QUESTION_CONFIG,
        split=split,
        streaming=streaming,
        revision=revision,
        token=token,
        **kwargs,
    )


def load_enterprise_rag(
    *,
    split: str = "test",
    streaming: bool = True,
    revision: str | None = None,
    token: str | bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load the document and question configurations with identical options."""
    common = {
        "split": split,
        "streaming": streaming,
        "revision": revision,
        "token": token,
        **kwargs,
    }
    return {
        "documents": load_enterprise_rag_documents(**common),
        "questions": load_enterprise_rag_questions(**common),
    }


__all__ = [
    "DATASET_ID",
    "DOCUMENT_CONFIG",
    "QUESTION_CONFIG",
    "load_enterprise_rag",
    "load_enterprise_rag_documents",
    "load_enterprise_rag_questions",
]
