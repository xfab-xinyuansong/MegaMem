from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import fingerprint, public_config
from .data import load_questions, load_records, select_questions
from .evaluation import Evaluator, aggregate, resolve_adapter
from .protocols import make_specs, resolved_config


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest() -> str:
    root = Path(__file__).resolve().parent
    paths = [*sorted(root.glob("*.py")), root.parent / "core" / "general_api.py"]
    return fingerprint({str(path.relative_to(root.parent)): file_digest(path) for path in paths})


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_path(directory: Path, question_id: str) -> Path:
    return directory / "records" / (fingerprint(question_id) + ".json")


def require_services(config: dict, embedding: bool) -> None:
    settings = config["runtime"]
    base = settings.get("api_base") or os.getenv(settings.get("api_base_env", "LLM_API_BASE"))
    secret = settings.get("api_key") or os.getenv(settings.get("api_key_env", "LLM_API_KEY"))
    if not base or not secret:
        raise ValueError("Configure the chat service through LLM_API_BASE and LLM_API_KEY or the selected environment variable names")
    if embedding:
        embedding_base = settings.get("embedding_api_base") or os.getenv(settings.get("embedding_api_base_env", "EMBEDDING_API_BASE")) or base
        embedding_secret = settings.get("embedding_api_key") or os.getenv(settings.get("embedding_api_key_env", "EMBEDDING_API_KEY")) or secret
        if not embedding_base or not embedding_secret:
            raise ValueError("Configure the embedding service credentials before execution")


def runtime_question(question: dict, gold: bool = False) -> dict:
    result = {key: copy.deepcopy(question[key]) for key in ("question_id", "question", "corpus_doc_ids") if key in question}
    if gold:
        if not isinstance(question.get("gold_chunks"), list):
            raise ValueError(f"Missing explicit gold_chunks for {question['question_id']}")
        result["gold_chunks"] = copy.deepcopy(question["gold_chunks"])
    return result


def paired_record(source: dict, variant: str, source_run: str) -> dict:
    result = copy.deepcopy(source)
    result.pop("metrics", None)
    result["paired_source_execution"] = {"timing": result.get("timing", {}), "usage": result.get("usage", {})}
    result["timing"] = {}
    result["usage"] = {}
    result["pairing"] = {
        "source_run": source_run,
        "variant": variant,
        "prediction_digest": fingerprint(source.get("prediction")),
        "additional_generation_calls": 0,
    }
    if variant == "no_attribution":
        result["reported_doc_ids"] = list(dict.fromkeys(result.get("loaded_doc_ids", [])))
    result.setdefault("trace", {})["paired_intervention"] = variant
    result["trace"]["reported_doc_ids"] = list(result.get("reported_doc_ids", []))
    return result


def paired_metrics(question: dict, record: dict, source_metrics: dict, evaluator: Evaluator) -> dict:
    if record["pairing"]["variant"] == "selective_oracle":
        return copy.deepcopy(source_metrics)
    metrics = evaluator.evaluate(question, record)
    for key in ("correctness", "completeness", "overall", "lexical_f1", "loaded_document_recall"):
        metrics[key] = source_metrics.get(key)
    if metrics["evaluation_status"] in {"scored", "partial", "failed"}:
        valid = sum(metrics.get(key) is not None for key in ("correctness", "completeness", "overall", "invalid_documents"))
        metrics["evaluation_status"] = "scored" if valid == 4 else "partial" if valid else "failed"
    metrics["answer_metrics_reused_from"] = record["pairing"]["source_run"]
    return metrics


def _questions(config: dict, spec) -> list[dict]:
    paths = config["paths"]
    rows = select_questions(
        load_questions(paths["questions"]), paths.get("split_manifest"), spec.split,
        spec.expected_count, spec.sample_size, config["runtime"]["seed"],
    )
    if spec.corpus_scope == "question_local":
        for row in rows:
            scope = row.get("corpus_doc_ids")
            if not isinstance(scope, list) or not scope or any(not isinstance(value, str) or not value.strip() for value in scope) or len(scope) != len(set(scope)):
                raise ValueError(f"{spec.dataset}:{row['question_id']} requires distinct nonempty corpus_doc_ids for question-local retrieval")
    if spec.variant == "selective_oracle" and any(not isinstance(row.get("answerable"), bool) for row in rows):
        raise ValueError("Selective oracle requires an explicit boolean answerable label for every question")
    if config["retrieval"]["evidence_source"] == "gold":
        if paths.get("gold_evidence"):
            by_id = {}
            for row in load_records(paths["gold_evidence"]):
                identifier = row.get("question_id")
                if identifier is None or isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                    raise ValueError("Gold evidence records require question_id")
                identifier = str(identifier)
                if identifier in by_id or not isinstance(row.get("gold_chunks"), list):
                    raise ValueError(f"Duplicate or malformed gold evidence: {identifier}")
                by_id[identifier] = row["gold_chunks"]
            for row in rows:
                if row["question_id"] not in by_id:
                    raise ValueError(f"Gold evidence missing question {row['question_id']}")
                row["gold_chunks"] = by_id[row["question_id"]]
        for row in rows:
            runtime_question(row, gold=True)
    return rows


def plan(config: dict, suites: list[str]) -> dict:
    runs = []
    for spec in make_specs(config, suites):
        resolved = resolved_config(config, spec)
        runs.append({**resolved["protocol"], "paths": resolved["paths"], "retrieval": resolved["retrieval"]})
    return {
        "runs": runs,
        "execution_requested": False,
        "official_metrics_available": bool(config["evaluation"].get("adapter")),
        "required_configuration": [name for name, value in {
            "models.reranker": config["models"].get("reranker"),
            "evaluation.adapter (optional; official metrics remain null without it)": config["evaluation"].get("adapter"),
        }.items() if not value],
        "parameter_provenance": config.get("parameter_provenance", {}),
    }


def build(config: dict, suites: list[str]) -> list[dict]:
    from .runtime import PaperRuntime

    pending = {}
    for spec in make_specs(config, suites):
        resolved = resolved_config(config, spec)
        path = resolved["paths"]["documents_template"]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        index_path = resolved["index"]["path"]
        if index_path in pending:
            previous = pending[index_path]
            if previous["index"] != resolved["index"] or previous["paths"]["documents_template"] != path:
                raise ValueError(f"Different corpora or extractors resolve to the same index path: {index_path}")
        pending.setdefault(index_path, resolved)
    results = []
    for resolved in pending.values():
        require_services(resolved, embedding=True)
    for resolved in pending.values():
        runtime = PaperRuntime(resolved)
        try:
            result = runtime.build(load_records(resolved["paths"]["documents_template"]))
        finally:
            runtime.close()
        results.append({"index": resolved["index"]["path"], "manifest": result})
    return results


def prepare_run(config: dict, spec, resume: bool) -> dict:
    from .build import build_identity
    from .backend import digest
    from .runtime import PaperRuntime

    resolved = resolved_config(config, spec)
    runtime = PaperRuntime(resolved)
    resolved = runtime.config
    questions = _questions(resolved, spec)
    index_manifest = None
    if resolved["retrieval"]["evidence_source"] != "gold":
        source = Path(resolved["index"]["path"]) / "manifest.json"
        index_manifest = json.loads(source.read_text())
        if index_manifest.get("status") != "complete":
            raise ValueError(f"Incomplete index: {source}")
        if index_manifest.get("fingerprint") != digest(build_identity(resolved)):
            raise ValueError(f"Index configuration mismatch: {source}")
        if not (source.parent / "sources.sqlite3").is_file():
            raise FileNotFoundError(source.parent / "sources.sqlite3")
    if resolved["retrieval"]["rerank"] and not resolved["models"].get("reranker") and resolved["retrieval"]["evidence_source"] != "gold" and not spec.paired_from:
        raise ValueError("Configure models.reranker explicitly before running retrieval")
    evaluator = Evaluator(resolved)
    adapter_digest = None
    if evaluator.adapter_spec:
        evaluator._adapter = resolve_adapter(evaluator.adapter_spec)
        adapter_source = inspect.getsourcefile(evaluator._adapter)
        if adapter_source and Path(adapter_source).is_file():
            adapter_digest = file_digest(adapter_source)
        elif not resolved["evaluation"].get("adapter_version"):
            raise ValueError("Set evaluation.adapter_version for an adapter without readable source")
    identity = {
        "format_version": 1,
        "configuration": public_config(resolved),
        "question_ids": [row["question_id"] for row in questions],
        "questions_digest": fingerprint(questions),
        "comparison_questions_digest": fingerprint([{key: row.get(key) for key in ("question_id", "question", "gold_answer", "expected_doc_ids", "answer_facts", "question_type")} for row in questions]),
        "evaluation_adapter_digest": adapter_digest,
        "index_manifest_digest": fingerprint(index_manifest),
        "implementation_digest": implementation_digest(),
    }
    manifest = {"fingerprint": fingerprint(identity), "identity": identity}
    directory = Path(resolved["paths"]["output_root"]) / spec.run_id
    existing = directory / "manifest.json"
    if existing.exists():
        if not resume:
            raise FileExistsError(f"Run already exists; use --resume after checking its configuration: {directory}")
        previous = json.loads(existing.read_text())
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(f"Cannot resume a changed protocol, dataset, implementation, or index: {directory}")
        manifest = previous
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing an output directory without a run manifest: {directory}")
    return {"spec": spec, "config": resolved, "runtime": runtime, "evaluator": evaluator,
            "questions": questions, "manifest": manifest, "directory": directory}


def summarize(directory: str | Path) -> dict:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest["identity"]["question_ids"]
    records = []
    for identifier in expected:
        path = record_path(root, identifier)
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("question_id") != identifier or record.get("run_fingerprint") != manifest["fingerprint"]:
                raise ValueError(f"Corrupt or mismatched record: {path}")
            records.append(record)
    result = {
        "run_id": manifest["identity"]["configuration"]["protocol"]["run_id"],
        "expected_count": len(expected), "missing_count": len(expected) - len(records),
        "run_fingerprint": manifest["fingerprint"], **aggregate(records),
    }
    atomic_json(root / "summary.json", result)
    predictions = root / "predictions.jsonl"
    temporary = predictions.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(predictions)
    temporary = root / "summary.csv.pending"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "metric", "value", "valid_count", "missing_or_invalid_count"])
        writer.writeheader()
        for metric, values in result["metrics"].items():
            writer.writerow({"run_id": result["run_id"], "metric": metric, **values})
    temporary.replace(root / "summary.csv")
    return result


def run(config: dict, suites: list[str], resume: bool = False) -> list[dict]:
    specs = make_specs(config, suites)
    prepared = [prepare_run(config, spec, resume) for spec in specs]
    question_sets = {}
    for item in prepared:
        spec = item["spec"]
        group = (spec.dataset, spec.split, spec.expected_count)
        signature = fingerprint([(row["question_id"], row["question"]) for row in item["questions"]])
        if group in question_sets and question_sets[group] != signature:
            raise ValueError(f"Experiments must share the same question set and order: {group}")
        question_sets[group] = signature
        if not spec.paired_from and any(not record_path(item["directory"], row["question_id"]).exists() for row in item["questions"]):
            require_services(item["config"], embedding=item["config"]["retrieval"]["evidence_source"] != "gold" and item["config"]["retrieval"]["backend"] != "bm25")
    completed = {}
    summaries = []
    for item in prepared:
        try:
            records = _execute_run(item, completed)
        finally:
            item["runtime"].close()
        completed[item["spec"].run_id] = records
        summaries.append(summarize(item["directory"]))
    return summaries


def _execute_run(item: dict, completed: dict) -> dict:
    spec = item["spec"]
    directory = item["directory"]
    manifest = item["manifest"]
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    atomic_json(directory / "manifest.json", manifest)
    records = {}
    for question in item["questions"]:
        identifier = question["question_id"]
        path = record_path(directory, identifier)
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("question_id") != identifier or record.get("run_fingerprint") != manifest["fingerprint"]:
                raise ValueError(f"Corrupt or mismatched record: {path}")
        else:
            if spec.paired_from:
                source_variant = spec.paired_from
                if spec.variant == "selective_oracle" and not question["answerable"]:
                    source_variant = "no_expansion"
                source_id = replace(spec, variant=source_variant).run_id
                source = completed[source_id][identifier]
                record = paired_record(source, spec.variant, source_id)
            else:
                try:
                    record = item["runtime"].answer(runtime_question(question, item["config"]["retrieval"]["evidence_source"] == "gold"))
                except Exception as exc:
                    record = {"question_id": identifier, "prediction": None, "status": "failed",
                              "errors": [f"{type(exc).__name__}: {exc}"], "timing": {}, "usage": {}}
            if record.get("question_id") != identifier:
                raise ValueError(f"Runtime returned a different question_id for {identifier}")
            record["run_fingerprint"] = manifest["fingerprint"]
            record["protocol"] = item["config"]["protocol"]
            record["oracle"] = spec.oracle
            record["question_type"] = question.get("question_type")
            record["errors"] = record.get("errors", record.get("trace", {}).get("errors", []))
            record["measurements"] = {
                "evidence_tokens": record.get("trace", {}).get("evidence_tokens"),
                "answer_tokens": record.get("trace", {}).get("answer_tokens"),
                "loaded_document_count": len(record["loaded_doc_ids"]) if "loaded_doc_ids" in record else None,
                "retrieved_document_count": len(record["retrieved_doc_ids"]) if "retrieved_doc_ids" in record else None,
                "reported_document_count": len(record["reported_doc_ids"]) if "reported_doc_ids" in record else None,
            }
            if spec.paired_from:
                record["metrics"] = paired_metrics(question, record, source["metrics"], item["evaluator"])
            else:
                record["metrics"] = item["evaluator"].evaluate(question, record)
            atomic_json(path, record)
        records[identifier] = record
    return records

def compare(first_directory: str | Path, second_directory: str | Path) -> dict:
    first_root, second_root = Path(first_directory), Path(second_directory)
    first_manifest = json.loads((first_root / "manifest.json").read_text())
    second_manifest = json.loads((second_root / "manifest.json").read_text())
    identities = [manifest["identity"] for manifest in (first_manifest, second_manifest)]
    for key in ("question_ids", "comparison_questions_digest"):
        if identities[0][key] != identities[1][key]:
            raise ValueError(f"Paired comparison requires identical {key}")
    for key in ("score_scale", "adapter", "adapter_version"):
        values = [identity["configuration"]["evaluation"].get(key) for identity in identities]
        if values[0] != values[1]:
            raise ValueError(f"Paired comparison requires identical evaluation.{key}")
    if identities[0].get("evaluation_adapter_digest") != identities[1].get("evaluation_adapter_digest"):
        raise ValueError("Paired comparison requires the same evaluation adapter implementation")
    for key in ("dataset", "scale", "extractor", "split", "corpus_scope"):
        if identities[0]["configuration"]["protocol"][key] != identities[1]["configuration"]["protocol"][key]:
            raise ValueError(f"Paired comparison requires identical {key}")
    pairs = []
    for identifier in identities[0]["question_ids"]:
        pair = []
        for root, manifest in ((first_root, first_manifest), (second_root, second_manifest)):
            record = json.loads(record_path(root, identifier).read_text())
            if record.get("question_id") != identifier or record.get("run_fingerprint") != manifest["fingerprint"]:
                raise ValueError("Paired comparison found a mismatched record")
            pair.append(record)
        pairs.append(pair)
    changes = {}
    for metric in ("correctness", "completeness", "overall", "invalid_documents", "loaded_document_recall", "reported_document_recall"):
        deltas = [right["metrics"][metric] - left["metrics"][metric] for left, right in pairs
                  if all(isinstance(row.get("metrics", {}).get(metric), (int, float)) and not isinstance(row["metrics"][metric], bool) and math.isfinite(row["metrics"][metric]) for row in (left, right))]
        changes[metric] = {"mean_difference_second_minus_first": sum(deltas) / len(deltas) if deltas else None,
                           "paired_valid_count": len(deltas), "missing_count": len(pairs) - len(deltas)}
    return {"first": str(first_root), "second": str(second_root), "count": len(pairs), "metric_changes": changes,
            "identical_answers": sum(left.get("prediction") == right.get("prediction") for left, right in pairs),
            "recovered_questions_with_any_gold_document": sum(left.get("metrics", {}).get("loaded_document_recall") == 0 and (right.get("metrics", {}).get("loaded_document_recall") or 0) > 0 for left, right in pairs),
            "lost_questions_with_any_gold_document": sum((left.get("metrics", {}).get("loaded_document_recall") or 0) > 0 and right.get("metrics", {}).get("loaded_document_recall") == 0 for left, right in pairs)}
