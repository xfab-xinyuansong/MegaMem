"""Installable dual-view memory components used by the research artifact."""

from importlib import import_module
from typing import Any

__all__ = [
    "DualNode",
    "DualNodeError",
    "TokenLedger",
    "read_nodes_jsonl",
    "validate_batch",
    "validate_one",
    "write_nodes_jsonl",
]

_EXPORTS = {
    "DualIndex": ("dual_index", "DualIndex"),
    "DualNode": ("dual_node", "DualNode"),
    "DualNodeError": ("dual_node", "DualNodeError"),
    "TokenLedger": ("token_ledger", "TokenLedger"),
    "build_l0_dualnodes": ("hierarchy_builder", "build_l0_dualnodes"),
    "read_nodes_jsonl": ("dual_node", "read_nodes_jsonl"),
    "validate_batch": ("dual_node", "validate_batch"),
    "validate_one": ("dual_node", "validate_one"),
    "write_nodes_jsonl": ("dual_node", "write_nodes_jsonl"),
}


def __getattr__(name: str) -> Any:
    """Lazily load components with optional model or vector-store dependencies."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List core and optional method components without importing backends."""
    return sorted(set(globals()) | set(_EXPORTS))
