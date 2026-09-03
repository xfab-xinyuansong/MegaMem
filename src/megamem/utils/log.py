import logging
from typing import Any, Dict, Optional

from megamem.core.memory_entry import MemoryEntry
from megamem.core.segment import Segment

logger = logging.getLogger(__name__)


def log_segments(segments: Segment) -> None:
    """Emit a single info log line listing each segment's heading.

    Args:
        segments: iterable of segments to summarise.
    """
    logger.info("### Logging segments:")
    body = ""
    for pos, seg in enumerate(segments):
        heading = seg.metadata.get("heading_path", f"Segment {pos + 1}")
        body += heading + "\n"
    logger.info(f"\n{body}")


def configure_logging(log_level: str = "INFO", log_dir: Optional[str] = None):
    """Configure root logging for the application.

    Both a console handler (always) and an optional file handler (when
    ``log_dir`` is supplied) are installed. A few noisy third-party
    libraries are also dialled down to WARNING.

    Args:
        log_level: textual logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_dir: when provided, a timestamped ``run_<ts>.log`` file is
            written under this directory (created if missing). The console
            handler is installed in addition, so output keeps appearing in
            the terminal.

    Returns:
        Path to the created log file, or ``None`` when no ``log_dir`` was
        given.
    """
    import os
    from datetime import datetime

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=fmt, force=True)
    logging.getLogger().setLevel(level)

    log_file_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"run_{ts}.log")
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(file_handler)

    quiet_loggers = (
        "requests",
        "urllib3",
        "chromadb.telemetry.product.posthog",
    )
    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file_path


def log_memory_building(context: str, user_id: str) -> None:

    logger.info(
        f"\n" + "-" * 60 + "\n"
        f"Building Memory - [{user_id}]\n{context}"
        f"\n" + "-" * 60
    )


def log_memory_operation(
    operation_type: str,
    entry: MemoryEntry,
    user_id: str,
) -> None:
    """Emit a uniformly formatted log entry for a memory-store operation.

    Centralising the formatting here keeps add/update/delete log output
    consistent across the codebase and makes downstream parsing easier.

    The emitted block contains a horizontal-rule separator, the operation
    type along with timestamp and user, and the entry's index/value/cue
    fields.

    Args:
        operation_type: human label for the operation (Add, Update, ...).
        entry: the ``MemoryEntry`` being processed.
        user_id: identifier of the user owning the entry.
    """
    log_message = (
        "\n" + "-" * 60 + "\n"
        f"MEMORY STORE: {operation_type}|{entry.creation_time}|{user_id}\n"
        f"Index: {entry.index}\n"
        f"Value: {entry.value}\n"
        f"Timestamp: {entry.timestamp}\n"
        f"cue indices: {entry.cue_indices}\n"
    )

    log_message += "-" * 60

    logger.info(log_message)


# def memory_store_log_add(logger, index: str, value: str, metadata: dict):
#     logger = logging.getLogger(__name__)
#     logger.info(
#         "\n" + "=" * 60 + "\n"
#         f"MEMORY STORE: Add|{metadata.get('timestamp', 'N/A')}|{metadata.get('user_id', 'N/A')}\n"
#         f"Index: {index}\n"
#         f"Value: {value}\n" + "=" * 60
#     )


# def memory_store_log_update(
#     logger, old_index: str, new_index: str, new_value: str, metadata: dict
# ):
#     logger = logging.getLogger(__name__)
#     logger.info(
#         "\n" + "=" * 60 + "\n"
#         f"MEMORY STORE: Update|{metadata.get('timestamp', 'N/A')}|{metadata.get('user_id', 'N/A')}\n"
#         f"Index: {old_index} -> {new_index}\n"
#         f"Update Value: {new_value}\n" + "=" * 60
#     )
