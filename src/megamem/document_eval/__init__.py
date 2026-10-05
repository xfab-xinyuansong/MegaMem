from megamem.document_eval.types import (
    RawChunkEntry,
    DistilledMemoryEntry,
    CognitiveEntry,
    SectionNode,
    DocumentNode,
    DocumentRetrievalConfig,
)
from importlib import import_module

__all__ = [
    "RawChunkEntry",
    "DistilledMemoryEntry",
    "CognitiveEntry",
    "SectionNode",
    "DocumentNode",
    "DocumentRetrievalConfig",
    "DocumentBuildPipeline",
    "DocumentRetriever",
]


def __getattr__(name):
    modules = {"DocumentBuildPipeline": "pipeline", "DocumentRetriever": "retriever"}
    if name in modules:
        value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
