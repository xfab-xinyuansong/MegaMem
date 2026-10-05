from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def _record(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: expected a JSON object")
    return dict(value)


def load_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with source.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}:{number}: invalid JSON") from exc
                    yield _record(value, f"{source}:{number}")
        return
    if suffix == ".json":
        with source.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if isinstance(data, Mapping):
            containers = [key for key in ("records", "questions", "data") if isinstance(data.get(key), list)]
            if len(containers) > 1:
                raise ValueError(f"{source}: ambiguous record containers {containers}")
            data = data[containers[0]] if containers else [data]
        if not isinstance(data, list):
            raise ValueError(f"{source}: expected an object or list of objects")
        for number, value in enumerate(data, 1):
            yield _record(value, f"{source}:record {number}")
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Reading parquet records requires the optional pyarrow dependency") from exc
        with parquet.ParquetFile(source) as parquet_file:
            for batch in parquet_file.iter_batches(batch_size=1024):
                for value in batch.to_pylist():
                    yield _record(value, str(source))
        return
    raise ValueError(f"Unsupported record format: {source.suffix}")


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _identifier(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label}: expected a nonempty string or integer identifier")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label}: empty identifier")
    return result


def _identifiers(value: Any, label: str, optional: bool = False) -> list[str] | None:
    if value is None and optional:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected a list of identifiers")
    result = [_identifier(item, label) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{label}: duplicate identifiers")
    return result


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    questions = []
    seen: set[str] = set()
    for position, row in enumerate(load_records(path), 1):
        label = f"{path}:question {position}"
        question_id = _identifier(_first(row, ("question_id", "query_id", "id")), label)
        if question_id in seen:
            raise ValueError(f"{label}: duplicate question_id {question_id!r}")
        seen.add(question_id)
        question = _first(row, ("question", "query", "query_text"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{label}: question text must be a nonempty string")
        gold = _first(row, ("gold_answer", "reference_answer", "reference", "answer"))
        if gold is not None and not isinstance(gold, str):
            raise ValueError(f"{label}: gold_answer must be a string or null")
        expected = _identifiers(
            _first(row, ("expected_doc_ids", "gold_doc_ids", "gold_document_ids")),
            f"{label}:expected_doc_ids",
            optional=True,
        )
        facts = _first(row, ("answer_facts", "gold_facts"))
        if facts is not None and (not isinstance(facts, list) or any(not isinstance(item, str) for item in facts)):
            raise ValueError(f"{label}: answer_facts must be a list of strings or null")
        chunks = row.get("gold_chunks")
        if chunks is not None and (not isinstance(chunks, list) or any(not isinstance(item, Mapping) for item in chunks)):
            raise ValueError(f"{label}: gold_chunks must be a list of objects or null")
        question_type = _first(row, ("question_type", "query_type", "type"))
        if question_type is not None and not isinstance(question_type, str):
            raise ValueError(f"{label}: question_type must be a string or null")
        questions.append({
            **row,
            "question_id": question_id,
            "question": question,
            "gold_answer": gold,
            "expected_doc_ids": expected,
            "answer_facts": facts,
            "question_type": question_type,
            "gold_chunks": [dict(chunk) for chunk in chunks] if chunks is not None else None,
        })
    return questions


def _manifest_ids(manifest: Mapping[str, Any], names: Sequence[str], label: str) -> list[str]:
    available = [name for name in names if name in manifest]
    if not available:
        raise ValueError(f"Split manifest is missing {label}")
    values = [_identifiers(manifest[name], f"manifest:{name}") for name in available]
    if any(set(value) != set(values[0]) for value in values[1:]):
        raise ValueError(f"Conflicting manifest aliases for {label}")
    return values[0]


def select_questions(
    questions: Sequence[Mapping[str, Any]],
    split_path: str | Path | None,
    split: str,
    expected_count: int,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    by_id: dict[str, dict[str, Any]] = {}
    for row in questions:
        question_id = _identifier(row.get("question_id"), "question_id")
        if question_id in by_id:
            raise ValueError(f"Duplicate question_id {question_id!r}")
        by_id[question_id] = {**row, "question_id": question_id}
    normalized_split = {"test": "validation", "val": "validation"}.get(split, split)
    if split_path is None:
        if normalized_split not in {"all", "task"}:
            raise ValueError("A split manifest is required for development or validation selection")
        selected_ids = list(by_id)
    else:
        with Path(split_path).open(encoding="utf-8-sig") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, Mapping):
            raise ValueError("Split manifest must be an object")
        if "seed" in manifest and (isinstance(manifest["seed"], bool) or manifest["seed"] != 42):
            raise ValueError("EnterpriseRAG split manifest must declare seed 42")
        dev = _manifest_ids(manifest, ("dev_question_ids", "development_question_ids", "dev"), "development IDs")
        validation = _manifest_ids(manifest, ("validation_question_ids", "test_question_ids", "validation", "test"), "validation IDs")
        if len(dev) != 100 or len(validation) != 400:
            raise ValueError(f"EnterpriseRAG requires 100 development and 400 validation IDs, found {len(dev)} and {len(validation)}")
        if set(dev) & set(validation):
            raise ValueError("Development and validation IDs overlap")
        manifest_ids = set(dev) | set(validation)
        if manifest_ids != set(by_id):
            missing = sorted(manifest_ids - set(by_id))
            unexpected = sorted(set(by_id) - manifest_ids)
            raise ValueError(f"Question IDs do not match manifest: missing={missing[:5]}, unexpected={unexpected[:5]}")
        if normalized_split == "dev":
            selected_ids = dev
        elif normalized_split == "validation":
            selected_ids = validation
        elif normalized_split == "all":
            selected_ids = dev + validation
        else:
            raise ValueError(f"Unknown EnterpriseRAG split {split!r}")
    if sample_size is not None:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if sample_size > len(selected_ids):
            raise ValueError(f"Cannot sample {sample_size} from {len(selected_ids)} questions")
        selected_ids = random.Random(seed).sample(selected_ids, sample_size)
    if len(selected_ids) != expected_count:
        raise ValueError(f"Selected {len(selected_ids)} questions for {split!r}; expected exactly {expected_count}")
    return [by_id[question_id] for question_id in selected_ids]
