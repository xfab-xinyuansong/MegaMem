from __future__ import annotations

import copy
import importlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from statistics import mean
from typing import Any


OFFICIAL_METRICS = ("correctness", "completeness", "overall", "invalid_documents")
PROXY_METRICS = ("lexical_f1", "loaded_document_recall", "reported_document_recall")
FAILED_STATUSES = {"failed", "failure", "error", "timeout", "invalid", "generation_failed", "retrieval_failed", "answer_failed"}


def resolve_adapter(spec: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("Evaluation adapter must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute or any(not part.isidentifier() for part in module_name.split(".") + attribute.split(".")):
        raise ValueError("Evaluation adapter must use module:function syntax")
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Evaluation adapter {spec!r} is not callable")
    return target


def _finite(value: Any, lower: float = 0.0, upper: float | None = None) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= lower
        and (upper is None or value <= upper)
    )


def lexical_f1(prediction: Any, gold: Any) -> float | None:
    if not isinstance(prediction, str) or not isinstance(gold, str) or not gold.strip():
        return None
    predicted = Counter(re.findall(r"\w+", prediction.casefold(), flags=re.UNICODE))
    expected = Counter(re.findall(r"\w+", gold.casefold(), flags=re.UNICODE))
    if not expected:
        return None
    overlap = sum((predicted & expected).values())
    if not overlap:
        return 0.0
    return 2.0 * overlap / (sum(predicted.values()) + sum(expected.values()))


def _id_set(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple, set)):
        return None
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) or not str(item).strip() for item in value):
        return None
    return {str(item).strip() for item in value}


def document_recall(observed: Any, expected: Any) -> float | None:
    gold = _id_set(expected)
    actual = _id_set(observed)
    if not gold or actual is None:
        return None
    return len(actual & gold) / len(gold)


def _runtime_failed(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "unknown")).lower()
    return status in FAILED_STATUSES or status.endswith("_failed")


class Evaluator:
    def __init__(self, config: Mapping[str, Any]):
        settings = config.get("evaluation", {})
        if not isinstance(settings, Mapping):
            raise ValueError("evaluation must be an object")
        self.adapter_spec = settings.get("adapter")
        self.score_scale = settings.get("score_scale", 100)
        if not _finite(self.score_scale) or self.score_scale == 0:
            raise ValueError("evaluation.score_scale must be positive and finite")
        if not isinstance(settings.get("allow_unscored", True), bool):
            raise ValueError("evaluation.allow_unscored must be boolean")
        self.allow_unscored = settings.get("allow_unscored", True)
        if self.adapter_spec is not None and not isinstance(self.adapter_spec, str):
            raise ValueError("evaluation.adapter must be module:function or null")
        if self.adapter_spec is None and not self.allow_unscored:
            raise ValueError("An evaluation adapter is required when allow_unscored is false")
        self._adapter = None

    def evaluate(self, question: Mapping[str, Any], prediction_record: Mapping[str, Any]) -> dict[str, Any]:
        failed = _runtime_failed(prediction_record)
        output: dict[str, Any] = {name: None for name in OFFICIAL_METRICS}
        output.update({
            "lexical_f1": None if failed else lexical_f1(prediction_record.get("prediction"), question.get("gold_answer")),
            "loaded_document_recall": document_recall(prediction_record.get("loaded_doc_ids"), question.get("expected_doc_ids")),
            "reported_document_recall": document_recall(prediction_record.get("reported_doc_ids"), question.get("expected_doc_ids")),
            "question_type": question.get("question_type"),
            "score_scale": float(self.score_scale),
            "evaluation_status": "unscored",
            "evaluation_errors": [],
            "adapter": self.adapter_spec,
        })
        if not question.get("question_id") or question.get("question_id") != prediction_record.get("question_id"):
            for name in PROXY_METRICS:
                output[name] = None
            output["evaluation_status"] = "failed"
            output["evaluation_errors"].append("Question and prediction question_id do not match")
            return output
        if failed:
            trace = prediction_record.get("trace", {})
            packed = isinstance(trace, Mapping) and (
                isinstance(trace.get("loaded_chunk_ids"), list) or _finite(trace.get("evidence_tokens"))
            )
            if not packed:
                output["loaded_document_recall"] = None
                output["reported_document_recall"] = None
            output["evaluation_status"] = "runtime_failed"
            return output
        if self.adapter_spec is None:
            return output
        if not isinstance(prediction_record.get("prediction"), str):
            output["evaluation_status"] = "failed"
            output["evaluation_errors"].append("prediction must be a string for adapter evaluation")
            return output
        try:
            if self._adapter is None:
                self._adapter = resolve_adapter(self.adapter_spec)
            values = self._adapter(copy.deepcopy(dict(question)), copy.deepcopy(dict(prediction_record)))
            if not isinstance(values, Mapping):
                raise TypeError("Evaluation adapter must return an object")
        except Exception as exc:
            output["evaluation_status"] = "failed"
            output["evaluation_errors"].append(f"{type(exc).__name__}: {exc}")
            return output
        for name in OFFICIAL_METRICS:
            value = values.get(name)
            upper = 1.0 if name == "invalid_documents" else float(self.score_scale)
            if not _finite(value, upper=upper):
                output["evaluation_errors"].append(f"{name}: expected a finite number in [0, {upper:g}]")
            else:
                output[name] = float(value)
        valid_count = sum(output[name] is not None for name in OFFICIAL_METRICS)
        output["evaluation_status"] = "scored" if valid_count == len(OFFICIAL_METRICS) else "partial" if valid_count else "failed"
        return output


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float], count: int) -> dict[str, Any]:
    return {
        "mean": mean(values) if values else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "total": sum(values) if values else None,
        "valid_count": len(values),
        "missing_or_invalid_count": count - len(values),
    }


def _numeric_leaves(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            result.update(_numeric_leaves(child, path))
        else:
            result[path] = child
    return result


def _measurement_stats(records: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        source = record.get(field)
        if not isinstance(source, Mapping):
            continue
        for key, value in _numeric_leaves(source).items():
            values.setdefault(key, [])
            if _finite(value):
                values[key].append(float(value))
    return {key: _stats(items, len(records)) for key, items in sorted(values.items())}


def _summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    summaries = {}
    for name in OFFICIAL_METRICS + PROXY_METRICS:
        valid = []
        for record in records:
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            if name in OFFICIAL_METRICS and (metrics.get("evaluation_status") not in {"scored", "partial"} or _runtime_failed(record)):
                continue
            if name == "lexical_f1" and _runtime_failed(record):
                continue
            value = metrics.get(name)
            scale = metrics.get("score_scale", 100)
            upper = scale if name in OFFICIAL_METRICS and name != "invalid_documents" else 1.0
            if _finite(upper) and _finite(value, upper=float(upper)):
                valid.append(float(value))
        summaries[name] = {"value": mean(valid) if valid else None, "valid_count": len(valid), "missing_or_invalid_count": count - len(valid)}
    runtime_status = Counter(str(record.get("status", "unknown")) for record in records)
    evaluation_status = Counter(
        str(record.get("metrics", {}).get("evaluation_status", "missing"))
        if isinstance(record.get("metrics"), Mapping) else "missing"
        for record in records
    )
    reported_counts = []
    for record in records:
        reported = _id_set(record.get("reported_doc_ids"))
        if reported is not None:
            reported_counts.append(float(len(reported)))
    scales = sorted({float(record["metrics"]["score_scale"]) for record in records if isinstance(record.get("metrics"), Mapping) and _finite(record["metrics"].get("score_scale"))})
    if len(scales) > 1:
        raise ValueError(f"Cannot aggregate official metrics with mixed score scales: {scales}")
    return {
        "count": count,
        "metrics": summaries,
        "score_scale": scales[0] if scales else None,
        "status_counts": dict(sorted(runtime_status.items())),
        "evaluation_status_counts": dict(sorted(evaluation_status.items())),
        "runtime_failure_count": sum(_runtime_failed(record) for record in records),
        "evaluation_failure_count": sum(evaluation_status.get(status, 0) for status in ("failed", "partial", "runtime_failed")),
        "records_with_errors": sum(bool(record.get("errors")) for record in records),
        "reported_documents": _stats(reported_counts, count),
        "timing": _measurement_stats(records, "timing"),
        "usage": _measurement_stats(records, "usage"),
        "evidence": _measurement_stats(records, "measurements"),
    }


def aggregate(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if any(not isinstance(record, Mapping) for record in rows):
        raise ValueError("Aggregate input must contain record objects")
    result = _summarize(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in rows:
        metrics = record.get("metrics", {})
        question_type = record.get("question_type")
        if question_type is None and isinstance(metrics, Mapping):
            question_type = metrics.get("question_type")
        grouped[str(question_type) if question_type else "unknown"].append(record)
    result["by_type"] = {kind: _summarize(group) for kind, group in sorted(grouped.items())}
    return result
