"""Latency tracking helpers for the memory pipeline.

Utilities here capture timing information across retrieval, formatting,
and answer generation so we can profile where time is spent.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Collects timing data for the memory retrieval pipeline.

    The tracker stores both raw per-operation timings and richer metadata
    such as multi-step retrieval traces and prompt statistics.

    Tracked dimensions:
      * Search latency (with breakdown by sub-component)
      * Per-step retrieval traces (for policy retrievers)
      * Memory formatting time
      * LLM generation time
      * Prompt token / item statistics

    Example:
        tracker = LatencyTracker()
        with tracker.track("search_primary"):
            ...
        with tracker.track("search_cue"):
            ...

        info = tracker.get_summary()
        print(f"Total search time: {info['total_search_time']:.3f}s")
    """

    def __init__(self):
        """Set up empty internal storage."""
        self._timings: Dict[str, List[float]] = {}
        self._retrieval_steps: List[Dict[str, Any]] = []
        self._prompt_stats: Dict[str, Any] = {}
        self._overall_search_start: Optional[float] = None
        self._overall_search_time: Optional[float] = None

    @contextmanager
    def track(self, operation: str):
        """Context manager that records the wall-clock time of a block.

        Args:
            operation: label identifying the timed operation.

        Example:
            with tracker.track("search_primary"):
                results = memory.query(...)
        """
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            self._timings.setdefault(operation, []).append(dt)
            logger.debug(f"[LatencyTracker] {operation}: {dt:.4f}s")

    def start_overall_search(self):
        """Start the wall-clock timer for the whole search phase."""
        self._overall_search_start = time.time()

    def end_overall_search(self):
        """Stop the wall-clock timer for the whole search phase."""
        if self._overall_search_start is not None:
            self._overall_search_time = time.time() - self._overall_search_start
            logger.debug(
                f"[LatencyTracker] Overall search: {self._overall_search_time:.4f}s"
            )

    def add_timing(self, operation: str, duration: float):
        """Append a manually measured duration.

        Args:
            operation: label of the operation.
            duration: time in seconds.
        """
        self._timings.setdefault(operation, []).append(duration)

    def add_retrieval_step(self, step_data: Dict[str, Any]):
        """Record a per-step entry for policy-style retrievers.

        Args:
            step_data: keys include ``step``, ``action``, ``duration``,
                optional ``query``, ``memories_count`` and additional
                metadata.
        """
        self._retrieval_steps.append(step_data)

    def set_prompt_stats(self, stats: Dict[str, Any]):
        """Store prompt-level statistics.

        Args:
            stats: usually contains ``total_tokens``, ``num_memories``,
                ``avg_tokens`` and (for multimodal prompts) ``num_images``.
        """
        self._prompt_stats = stats

    def get_timing(self, operation: str) -> float:
        """Return the cumulative time recorded for an operation.

        Args:
            operation: label to look up.

        Returns:
            Sum (in seconds) of every timing recorded for that label.
        """
        return sum(self._timings.get(operation, []))

    def get_timing_list(self, operation: str) -> List[float]:
        """Return every individual measurement for an operation.

        Args:
            operation: label to look up.

        Returns:
            All measured durations (seconds), in insertion order.
        """
        return self._timings.get(operation, [])

    def get_search_breakdown(self) -> Dict[str, float]:
        """Return per-component cumulative search timings.

        Returns:
            Mapping from search component name to total seconds spent.
        """
        component_keys = (
            "search_primary",
            "search_cue",
            "search_hybrid",
            "search_rrf_merge",
            "search_llm_filter",
            "search_keyword_extract",
            "search_bm25_index_build",
        )
        return {
            key: sum(self._timings[key])
            for key in component_keys
            if key in self._timings
        }

    def get_retrieval_steps_summary(self) -> Dict[str, Any]:
        """Aggregate per-step retrieval information.

        Returns:
            Empty dict when no steps recorded; otherwise a dict with
            ``num_steps``, ``steps`` and ``total_step_time``.
        """
        if not self._retrieval_steps:
            return {}

        total_step_time = sum(
            step.get("duration", 0) for step in self._retrieval_steps
        )

        return {
            "num_steps": len(self._retrieval_steps),
            "steps": self._retrieval_steps,
            "total_step_time": round(total_step_time, 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Build the full latency snapshot.

        Returns:
            Dictionary with combined timing/statistics fields. Possible
            keys include ``total_search_time``, ``search_breakdown``,
            ``search_breakdown_pct``, ``retrieval_steps``, ``format_time``,
            ``llm_time``, ``prompt_stats`` and ``total_time``.
        """
        out: Dict[str, Any] = {}

        if self._overall_search_time is not None:
            out["total_search_time"] = round(self._overall_search_time, 4)

        breakdown = self.get_search_breakdown()
        if breakdown:
            out["search_breakdown"] = {k: round(v, 4) for k, v in breakdown.items()}

            if self._overall_search_time and self._overall_search_time > 0:
                covered = sum(breakdown.values())
                out["search_breakdown_pct"] = {
                    k: round(v / self._overall_search_time * 100, 2)
                    for k, v in breakdown.items()
                }

                leftover = self._overall_search_time - covered
                if leftover > 0.001:
                    out["search_breakdown"]["search_unaccounted"] = round(leftover, 4)
                    out["search_breakdown_pct"]["search_unaccounted"] = round(
                        leftover / self._overall_search_time * 100, 2
                    )

        steps_summary = self.get_retrieval_steps_summary()
        if steps_summary:
            out["retrieval_steps"] = steps_summary

        format_time = self.get_timing("format_memories")
        if format_time > 0:
            out["format_time"] = round(format_time, 4)

        llm_time = self.get_timing("llm_generation")
        if llm_time > 0:
            out["llm_time"] = round(llm_time, 4)

        if self._prompt_stats:
            out["prompt_stats"] = self._prompt_stats

        total_time = 0.0
        if self._overall_search_time:
            total_time += self._overall_search_time
        if format_time > 0:
            total_time += format_time
        if llm_time > 0:
            total_time += llm_time

        if total_time > 0:
            out["total_time"] = round(total_time, 4)

        return out

    def log_summary(self, level: int = logging.INFO):
        """Emit the latency summary to the module logger.

        Args:
            level: logging level used for every emitted line (default INFO).
        """
        info = self.get_summary()

        logger.log(level, "=" * 60)
        logger.log(level, "Latency Summary")
        logger.log(level, "=" * 60)

        if "total_search_time" in info:
            logger.log(level, f"Total Search Time: {info['total_search_time']:.4f}s")

        if "search_breakdown" in info:
            logger.log(level, "\nSearch Breakdown:")
            for key, value in info["search_breakdown"].items():
                pct = info.get("search_breakdown_pct", {}).get(key, 0)
                logger.log(level, f"  {key}: {value:.4f}s ({pct:.1f}%)")

        if "retrieval_steps" in info:
            steps_info = info["retrieval_steps"]
            logger.log(level, f"\nRetrieval Steps: {steps_info['num_steps']}")
            logger.log(level, f"Total Step Time: {steps_info['total_step_time']:.4f}s")
            for step in steps_info["steps"]:
                action = step.get("action", "UNKNOWN")
                duration = step.get("duration", 0)
                step_num = step.get("step", "?")
                logger.log(level, f"  Step {step_num} ({action}): {duration:.4f}s")

        if "format_time" in info:
            logger.log(level, f"\nFormat Time: {info['format_time']:.4f}s")

        if "llm_time" in info:
            logger.log(level, f"LLM Generation Time: {info['llm_time']:.4f}s")

        if "prompt_stats" in info:
            logger.log(level, "\nPrompt Statistics:")
            for key, value in info["prompt_stats"].items():
                logger.log(level, f"  {key}: {value}")

        if "total_time" in info:
            logger.log(level, f"\nTotal Time: {info['total_time']:.4f}s")

        logger.log(level, "=" * 60)


# Memoised tokenizer encodings, keyed by model name.
_ENCODING_CACHE: Dict[str, Any] = {}


def get_encoding(model: str = "cl100k_base"):
    """Return (and lazily build) a cached tiktoken encoding.

    Falls back to the ``cl100k_base`` encoding when the model is unknown.

    Args:
        model: model identifier.

    Returns:
        A tiktoken ``Encoding`` instance.
    """
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Exact token count using tiktoken.

    Args:
        text: input string.
        model: model whose tokenizer to use.

    Returns:
        Number of tokens in ``text``.
    """
    return len(get_encoding(model).encode(text))


def count_memories_tokens(memories: List[str], model: str = "cl100k_base") -> Dict[str, int]:
    """Token statistics for a list of memory strings.

    Args:
        memories: memory strings to count.
        model: model whose tokenizer to use.

    Returns:
        Dict with ``total_tokens``, ``num_memories`` and ``avg_tokens``.
    """
    enc = get_encoding(model)
    total = sum(len(enc.encode(memory)) for memory in memories)
    avg = total / len(memories) if memories else 0

    return {
        "total_tokens": total,
        "num_memories": len(memories),
        "avg_tokens": round(avg, 2),
    }
