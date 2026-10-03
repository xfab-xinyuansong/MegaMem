import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


class LatencyTracker:

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._retrieval_steps: List[Dict[str, Any]] = []
        self._prompt_stats: Dict[str, Any] = {}
        self._overall_search_start: Optional[float] = None
        self._overall_search_time: Optional[float] = None

    @contextmanager
    def track(self, operation: str):
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            self._timings.setdefault(operation, []).append(dt)
            logger.debug(f"[LatencyTracker] {operation}: {dt:.4f}s")

    def start_overall_search(self):
        self._overall_search_start = time.time()

    def end_overall_search(self):
        if self._overall_search_start is not None:
            self._overall_search_time = time.time() - self._overall_search_start
            logger.debug(
                f"[LatencyTracker] Overall search: {self._overall_search_time:.4f}s"
            )

    def add_timing(self, operation: str, duration: float):
        self._timings.setdefault(operation, []).append(duration)

    def add_retrieval_step(self, step_data: Dict[str, Any]):
        self._retrieval_steps.append(step_data)

    def set_prompt_stats(self, stats: Dict[str, Any]):
        self._prompt_stats = stats

    def get_timing(self, operation: str) -> float:
        return sum(self._timings.get(operation, []))

    def get_timing_list(self, operation: str) -> List[float]:
        return self._timings.get(operation, [])

    def get_search_breakdown(self) -> Dict[str, float]:
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


_ENCODING_CACHE: Dict[str, Any] = {}


def get_encoding(model: str = "cl100k_base"):
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    return len(get_encoding(model).encode(text))


def count_memories_tokens(memories: List[str], model: str = "cl100k_base") -> Dict[str, int]:
    enc = get_encoding(model)
    total = sum(len(enc.encode(memory)) for memory in memories)
    avg = total / len(memories) if memories else 0

    return {
        "total_tokens": total,
        "num_memories": len(memories),
        "avg_tokens": round(avg, 2),
    }
