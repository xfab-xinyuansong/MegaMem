"""End-to-end document evaluation runner.

Loads authorized document tables, question streams, optional corpus manifests,
and optional split files. Builds the document-memory collections via
DocumentBuildPipeline, then runs retrieval, answer generation, and metric export.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def load_eval_inputs(
    docs_parquet: str,
    questions_jsonl: str,
    *,
    tier_manifest_parquet: Optional[str] = None,
    split_json: Optional[str] = None,
    split: str = "dev",
    max_questions: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load docs + questions, optionally restricted by tier and dev/test split."""
    import pandas as pd

    docs_df = pd.read_parquet(docs_parquet)
    # Optional tier restriction
    if tier_manifest_parquet:
        tier_df = pd.read_parquet(tier_manifest_parquet)
        tier_ids = set(tier_df["doc_id"].tolist())
        docs_df = docs_df[docs_df["doc_id"].isin(tier_ids)].copy()
        logger.info(f"Restricted to {len(docs_df)} docs from tier manifest")

    docs = []
    for _, row in docs_df.iterrows():
        docs.append({
            "doc_id": _safe_str(row.get("doc_id")),
            "title": _safe_str(row.get("title")),
            "source_type": _safe_str(row.get("source_type")),
            "content": _safe_str(row.get("content")),
        })

    # Load questions
    qs: List[Dict[str, Any]] = []
    with open(questions_jsonl) as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))

    # Dev/test split filter
    if split_json:
        with open(split_json) as f:
            split_data = json.load(f)
        ids_to_keep: Set[str]
        if split == "dev":
            ids_to_keep = set(split_data.get("dev_question_ids", []))
        elif split == "test":
            ids_to_keep = set(split_data.get("test_question_ids", []))
        elif split == "all":
            ids_to_keep = set(split_data.get("dev_question_ids", []) + split_data.get("test_question_ids", []))
        else:
            raise ValueError(f"unknown split: {split}")
        qs = [q for q in qs if q.get("question_id") in ids_to_keep]
        logger.info(f"After {split} split filter: {len(qs)} questions")

    if max_questions:
        qs = qs[:max_questions]

    return docs, qs


def run_eval(
    cfg,
    docs: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    *,
    method_name: str,
    output_dir: str,
    run_build: bool = True,
    build_distilled: bool = True,
    build_cognitive: bool = True,
    build_section_summaries: bool = True,
    build_document_summaries: bool = True,
    eval_workers: int = 4,
    skip_judge: bool = False,
) -> Dict[str, Any]:
    """Build + evaluate one method configuration."""
    from megamem.document_eval import DocumentBuildPipeline, DocumentRetriever
    from megamem.document_eval.answering import generate_answer
    from megamem.document_eval.metrics import (
        bleu_score, doc_recall, f1_score, llm_judge_score, text_recall,
    )
    from megamem.document_eval.storage import DocumentStorage

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    storage = DocumentStorage(cfg)

    build_stats: Dict[str, Any] = {}
    if run_build:
        pipeline = DocumentBuildPipeline(cfg, storage=storage)
        build_stats = pipeline.build(
            docs,
            build_distilled=build_distilled,
            build_cognitive=build_cognitive,
            build_section_summaries=build_section_summaries,
            build_document_summaries=build_document_summaries,
            max_extract_workers=eval_workers,
        )
        with open(os.path.join(output_dir, "build_stats.json"), "w") as f:
            json.dump(build_stats, f, indent=2)
        logger.info(f"Build done: {build_stats}")

    # Sanity counts
    collection_counts = {k: storage.count(k) for k in DocumentStorage.KINDS}
    logger.info(f"Collection counts: {collection_counts}")

    retriever = DocumentRetriever(cfg, storage=storage)

    # ------------- per-question eval (parallel) -------------
    per_question_records: List[Dict[str, Any]] = []
    t_search_eval = time.time()

    def _process_q(q: Dict[str, Any]) -> Dict[str, Any]:
        qid = q.get("question_id", "")
        question = q.get("question", "")
        gold = q.get("gold_answer", "")
        expected_docs = q.get("expected_doc_ids", []) or []
        answer_facts = q.get("answer_facts", []) or []
        qtype = q.get("question_type", "")
        # 1. retrieval
        result = retriever.retrieve(question)
        retrieved_docs = result.documents_retrieved
        # 2. answer
        ans_out = generate_answer(cfg, question, result.chunks)
        pred = ans_out["answer"]
        # 3. metrics
        evidence_text = "\n".join(c.get("raw_text", "") for c in result.chunks)
        bleu = bleu_score(pred, gold)
        f1 = f1_score(pred, gold)
        dr = doc_recall(retrieved_docs, expected_docs)
        tr = text_recall(answer_facts, evidence_text, fallback_gold=gold)
        if skip_judge:
            judge = {"score": 0, "reasoning": "skipped"}
        else:
            judge = llm_judge_score(cfg, question, gold, pred)

        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold,
            "expected_doc_ids": expected_docs,
            "retrieved_doc_ids": retrieved_docs,
            "retrieved_chunk_ids": [c["chunk_id"] for c in result.chunks],
            "prediction": pred,
            "primary_cognitive_types": result.primary_cognitive_types,
            "secondary_cognitive_types": result.secondary_cognitive_types,
            "retrieval_seconds": result.retrieval_seconds,
            "metrics": {
                "bleu_score": bleu,
                "f1_score": f1,
                "llm_score": int(judge["score"]),
                "doc_recall": dr,
                "text_recall": tr,
            },
            "judge_reasoning": judge.get("reasoning", ""),
        }

    with ThreadPoolExecutor(max_workers=eval_workers) as ex:
        futures = [ex.submit(_process_q, q) for q in questions]
        done = 0
        for fut in as_completed(futures):
            per_question_records.append(fut.result())
            done += 1
            if done % 10 == 0:
                logger.info(f"eval progress: {done}/{len(questions)}")

    t_search_eval = time.time() - t_search_eval

    # ------------- aggregate -------------
    n = len(per_question_records)
    if n == 0:
        agg: Dict[str, float] = {k: 0.0 for k in ["bleu_score", "f1_score", "llm_score", "doc_recall", "text_recall"]}
    else:
        agg = {
            k: round(sum(r["metrics"][k] for r in per_question_records) / n, 4)
            for k in ["bleu_score", "f1_score", "llm_score", "doc_recall", "text_recall"]
        }

    by_type: Dict[str, Dict[str, Any]] = {}
    type_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_question_records:
        type_buckets.setdefault(r["question_type"] or "unknown", []).append(r)
    for qt, bucket in type_buckets.items():
        nq = len(bucket)
        by_type[qt] = {
            "n": nq,
            **{
                k: round(sum(r["metrics"][k] for r in bucket) / nq, 4)
                for k in ["bleu_score", "f1_score", "llm_score", "doc_recall", "text_recall"]
            },
        }

    summary = {
        "n_questions": n,
        "aggregate": agg,
        "per_type": by_type,
        "wall_seconds_search_eval": round(t_search_eval, 1),
    }

    # ------------- write outputs -------------
    with open(os.path.join(output_dir, "per_question.json"), "w") as f:
        json.dump(per_question_records, f, indent=2)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Canonical results.json (project format, single experiment object)
    canonical = [
        {
            "project_id": "largecontextwindow",
            "experiment_id": f"stage1_mvb_0m_{method_name}",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": method_name,
            "description": "Stage 1 MVB at 0M tier (EnterpriseRAG gold-only subset).",
            "config": {
                "method_toggles": {
                    "enable_dual_index": cfg.enable_dual_index,
                    "enable_raw_stream": cfg.enable_raw_stream,
                    "enable_distilled_stream": cfg.enable_distilled_stream,
                    "enable_hierarchical": cfg.enable_hierarchical,
                    "document_routing_enabled": cfg.document_routing_enabled,
                    "section_routing_enabled": cfg.section_routing_enabled,
                    "enable_cdm": cfg.enable_cdm,
                    "enable_cognitive_path": cfg.enable_cognitive_path,
                    "relation_expansion_depth": cfg.relation_expansion_depth,
                },
                "retrieval_params": {
                    "K_A": cfg.K_A,
                    "K_B": cfg.K_B,
                    "alpha": cfg.alpha,
                    "K_D": cfg.K_D,
                    "K_S": cfg.K_S,
                    "primary_w": cfg.primary_weight,
                    "expansion_w": cfg.expansion_weight,
                    "top_n_final": cfg.top_n_final,
                    "llm_token_budget": cfg.llm_token_budget,
                },
                "build_params": {
                    "chunk_target_tokens": cfg.chunk_target_tokens,
                    "max_chunks_per_doc": cfg.max_chunks_per_doc,
                    "distilled_memory_per_chunk_budget": cfg.distilled_memory_per_chunk_budget,
                },
                "models": {
                    "chat": cfg.chat_model_id,
                    "judge": cfg.judge_model_id,
                    "embedding": cfg.local_embedding_model if cfg.use_local_embedding else "hosted",
                },
                "seed": cfg.seed,
                "dataset": "authorized document evaluation split",
            },
            "results": {
                "main_metric": {
                    "name": "llm_score",
                    "mean": agg["llm_score"],
                    "n_questions": n,
                },
                "five_metrics": agg,
                "per_type": by_type,
                "build_stats": build_stats,
                "collection_counts": collection_counts,
            },
            "timing": {
                "build_seconds": build_stats.get("build_seconds", 0.0),
                "extract_seconds": build_stats.get("extract_seconds", 0.0),
                "search_eval_seconds": round(t_search_eval, 1),
            },
        }
    ]
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(canonical, f, indent=2)

    logging.getLogger().removeHandler(fh)
    fh.close()
    logger.info(f"Eval complete: agg={agg}, per_type sizes={[(k, v['n']) for k, v in by_type.items()]}")
    return {"summary": summary, "canonical": canonical[0]}


# -------------------------------------------------------------------------
# CLI: cycle through 4 method configs (DDI / HDM / CDM / Combined)
# -------------------------------------------------------------------------


METHOD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ddi": {
        # DDI only: dual stream, no HDM routing, no CDM
        "enable_dual_index": True,
        "enable_raw_stream": True,
        "enable_distilled_stream": True,
        "enable_hierarchical": False,
        "document_routing_enabled": False,
        "section_routing_enabled": False,
        "enable_cdm": False,
        "enable_cognitive_path": False,
    },
    "hdm": {
        # HDM-only evaluation keeps routing off at 0M and retains hierarchical
        # scoring through document and section summary collections.
        "enable_dual_index": True,
        "enable_raw_stream": True,
        "enable_distilled_stream": False,
        "enable_hierarchical": True,
        "document_routing_enabled": False,
        "section_routing_enabled": False,
        "enable_cdm": False,
        "enable_cognitive_path": False,
    },
    "cdm": {
        # CDM only: cognitive path + raw fallback (so chunks still get retrieved)
        "enable_dual_index": True,
        "enable_raw_stream": True,
        "enable_distilled_stream": False,
        "enable_hierarchical": False,
        "document_routing_enabled": False,
        "section_routing_enabled": False,
        "enable_cdm": True,
        "enable_cognitive_path": True,
    },
    "combined": {
        # Option A full (Stage 1 0M: HDM routing OFF)
        "enable_dual_index": True,
        "enable_raw_stream": True,
        "enable_distilled_stream": True,
        "enable_hierarchical": True,
        "document_routing_enabled": False,
        "section_routing_enabled": False,
        "enable_cdm": True,
        "enable_cognitive_path": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-parquet", required=True)
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--tier-manifest", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["ddi", "hdm", "cdm", "combined"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chroma-root", required=True)
    parser.add_argument("--llm-api-base", default=os.getenv("LLM_API_BASE", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--chat-model", default=os.getenv("LLM_CHAT_MODEL", "YOUR_CHAT_MODEL"))
    parser.add_argument("--judge-model", default=os.getenv("LLM_JUDGE_MODEL", "YOUR_JUDGE_MODEL"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--build-only-once", action="store_true",
                        help="Build the common collections once for the first method, "
                             "then reuse for subsequent methods.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from megamem.document_eval.types import DocumentRetrievalConfig
    base_cfg_kwargs = dict(
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        chat_model_id=args.chat_model,
        judge_model_id=args.judge_model,
        chroma_path=args.chroma_root,
        collection_prefix="stage1_mvb",
        use_local_embedding=True,
    )

    docs, questions = load_eval_inputs(
        args.docs_parquet,
        args.questions_jsonl,
        tier_manifest_parquet=args.tier_manifest,
        split_json=args.split_json,
        split=args.split,
        max_questions=args.max_questions,
    )
    logger.info(f"Loaded {len(docs)} docs / {len(questions)} questions")

    for i, method in enumerate(args.methods):
        toggles = METHOD_CONFIGS[method]
        cfg = DocumentRetrievalConfig(**base_cfg_kwargs, **toggles)
        out_dir = os.path.join(args.output_root, method)
        run_build = True
        if args.build_only_once and i > 0:
            run_build = False
        logger.info(f"=== Running method={method} (build={run_build}) ===")
        run_eval(
            cfg,
            docs,
            questions,
            method_name=method,
            output_dir=out_dir,
            run_build=run_build,
            eval_workers=args.workers,
            skip_judge=args.skip_judge,
        )


if __name__ == "__main__":
    main()
