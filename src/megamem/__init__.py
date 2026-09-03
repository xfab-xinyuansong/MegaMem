"""Public package surface for MegaMem."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
# Star imports expose only interfaces supported by the core installation.
__all__ = [
    "DualNode",
    "DualNodeError",
    "GeneralAPIClient",
    "GeneralAPIError",
    "MemoryClient",
    "TokenLedger",
    "validate_batch",
    "validate_one",
    "__version__",
]

_EXPORTS = {
    "DualIndex": ("megamem.methods", "DualIndex"),
    "DualNode": ("megamem.methods", "DualNode"),
    "DualNodeError": ("megamem.methods", "DualNodeError"),
    "GeneralAPIClient": ("megamem.core.general_api", "GeneralAPIClient"),
    "GeneralAPIError": ("megamem.core.general_api", "GeneralAPIError"),
    "MemoryClient": ("megamem.client", "MemoryClient"),
    "TokenLedger": ("megamem.methods", "TokenLedger"),
    "validate_batch": ("megamem.methods", "validate_batch"),
    "validate_one": ("megamem.methods", "validate_one"),
}


def __getattr__(name: str) -> Any:
    """Load optional subpackages only when requested.

    Keeping the top-level import light makes version checks, packaging, and the
    representation-level tests independent of optional retrieval backends.
    """
    if name in {"browser", "methods", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List both core exports and lazily available optional interfaces."""
    return sorted(set(globals()) | set(_EXPORTS) | {"browser", "methods", "utils"})
