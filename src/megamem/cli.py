"""Command-line interface for MegaMem."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from importlib.util import find_spec
from typing import Any, Sequence

from . import __version__

logger = logging.getLogger(__name__)

_OPTIONAL_GROUPS = {
    "retrieval": (
        "chromadb",
        "faiss",
        "numpy",
        "rank_bm25",
        "sentence_transformers",
        "tiktoken",
    ),
    "llm": ("dotenv",),
    "local-models": ("accelerate", "torch", "transformers"),
    "documents": ("docx", "fitz", "markdownify", "openpyxl", "pdfplumber", "pptx"),
    "evaluation": (
        "bert_score",
        "nltk",
        "pandas",
        "pyarrow",
        "rouge_score",
        "scipy",
        "sklearn",
    ),
    "dev": ("build", "pytest", "pytest_timeout"),
}
_REQUIRED_MODEL_ALIASES = {"chat_low", "chat_high", "judge"}


def _show_config(_args: argparse.Namespace) -> int:
    """Print the active model config and its non-secret alias mapping."""
    from .methods.configs.model_resolver import list_aliases, models_config_path

    print(f"Model config: {models_config_path()}")
    for alias, status in sorted(list_aliases().items()):
        print(f"  {alias}: {status}")
    return 0


def _doctor_report() -> dict[str, Any]:
    """Collect local readiness information without making network requests."""
    from .methods.configs.model_resolver import list_aliases, models_config_path

    python_supported = sys.version_info >= (3, 11)
    config_path = ""
    config_error = ""
    aliases: dict[str, str] = {}
    try:
        config_path = str(models_config_path())
        aliases = list_aliases()
    except Exception as exc:  # diagnostics should report configuration failures
        config_error = str(exc)

    model_markers = ("PLACEHOLDER", "UNRESOLVED", "INVALID")
    models_ready = _REQUIRED_MODEL_ALIASES.issubset(aliases) and all(
        not any(marker in aliases[alias].upper() for marker in model_markers)
        for alias in _REQUIRED_MODEL_ALIASES
    )
    extras = {
        group: all(find_spec(module) is not None for module in modules)
        for group, modules in _OPTIONAL_GROUPS.items()
    }
    return {
        "package": "MegaMem",
        "version": __version__,
        "python": platform.python_version(),
        "python_supported": python_supported,
        "core_ready": python_supported and not config_error,
        "models_ready": models_ready,
        "model_config": config_path,
        "config_error": config_error or None,
        "aliases": aliases,
        "required_aliases": sorted(_REQUIRED_MODEL_ALIASES),
        "optional_groups": extras,
    }


def _run_doctor(args: argparse.Namespace) -> int:
    report = _doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"MegaMem {report['version']}")
        print(
            f"  Python {report['python']}: "
            f"{'ready' if report['python_supported'] else 'requires 3.11+'}"
        )
        print(f"  Core package: {'ready' if report['core_ready'] else 'not ready'}")
        print(f"  Model aliases: {'ready' if report['models_ready'] else 'not configured'}")
        print(f"  Model config: {report['model_config'] or 'unavailable'}")
        if report["config_error"]:
            print(f"  Config error: {report['config_error']}")
        print("  Optional groups:")
        for group, installed in report["optional_groups"].items():
            print(f"    {group}: {'installed' if installed else 'not installed'}")

    if args.strict and not (report["core_ready"] and report["models_ready"]):
        return 1
    return 0


def _dispatch_browser(args: argparse.Namespace) -> int:
    """Forward the browser subcommand to the dedicated module CLI."""
    from .browser.__main__ import main as browser_main

    forwarded: list[str] = [args.db_path]
    for attr, flag in (
        ("collection", "-c"),
        ("search", "--search"),
        ("output", "--output"),
    ):
        value = getattr(args, attr, None)
        if value:
            forwarded.extend([flag, value])

    if args.limit:
        forwarded.extend(["--limit", str(args.limit)])

    for attr, flag in (
        ("stats", "--stats"),
        ("analyze", "--analyze"),
        ("verbose", "--verbose"),
        ("no_interactive", "--no-interactive"),
    ):
        if getattr(args, attr, False):
            forwarded.append(flag)

    saved_argv = sys.argv
    try:
        sys.argv = ["megamem-browser", *forwarded]
        browser_main()
    finally:
        sys.argv = saved_argv
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaMem - source-resolved memory beyond the prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  megamem doctor
  megamem config
  megamem browser /path/to/memory_store --stats
  megamem browser /path/to/memory_store --search "incident management"
""",
    )
    parser.add_argument("--version", action="version", version=f"MegaMem {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check package, configuration, and optional dependencies"
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless the core package and every model alias are ready",
    )
    doctor_parser.set_defaults(func=_run_doctor)

    config_parser = subparsers.add_parser(
        "config", help="Show the active non-secret model-alias configuration"
    )
    config_parser.set_defaults(func=_show_config)

    browser_parser = subparsers.add_parser(
        "browser",
        help="Inspect a local memory store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    browser_parser.add_argument("db_path", help="Path to a ChromaDB directory")
    browser_parser.add_argument("--collection", "-c", help="Collection to inspect")
    browser_parser.add_argument("--stats", action="store_true", help="Show statistics and exit")
    browser_parser.add_argument("--analyze", action="store_true", help="Write an analysis report")
    browser_parser.add_argument("--search", "-s", help="Search query")
    browser_parser.add_argument("--output", "-o", help="Output path")
    browser_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum results")
    browser_parser.add_argument("--no-interactive", action="store_true", help="Disable prompts")
    browser_parser.set_defaults(func=_dispatch_browser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    from .utils.log import configure_logging

    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.debug("Command failed", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
