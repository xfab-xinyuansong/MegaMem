"""Option E (combined distilled+cognitive prompt) quality preflight."""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _per_chunk_extract_separate(cfg, chunk_meta: Dict[str, Any]) -> Tuple[list, list]:
    from megamem.document_eval.extractors import (
        extract_cognitive_relations,
        extract_distilled_memories,
    )
    d = extract_distilled_memories(cfg, **chunk_meta)
    c = extract_cognitive_relations(cfg, **chunk_meta)
    return d, c


def _per_chunk_extract_combined(cfg, chunk_meta: Dict[str, Any]) -> Tuple[list, list]:
    from megamem.document_eval.extractors import (
        extract_combined_distilled_and_cognitive,
    )
    return extract_combined_distilled_and_cognitive(cfg, **chunk_meta)


def run_preflight(
    docs: List[Dict[str, Any]],
    cfg,
    n_docs: int = 10,
) -> Dict[str, Any]:
    """For each of n_docs, segment and run both extractors on each chunk.

    Returns dict with per-mode counts + verdict.
    """
    from megamem.document_eval.chunking import segment_document

    docs = docs[:n_docs]
    logger.info(f"Option E preflight on {len(docs)} docs")

    # 1. Segment (deterministic, same chunks for both modes)
    all_chunk_meta: List[Dict[str, Any]] = []
    for d in docs:
        doc_id = d["doc_id"]
        title = (d.get("title") or "")[:300]
        source_type = d.get("source_type", "")
        content = d.get("content") or ""
        sections = segment_document(
            doc_id, content,
            target_tokens=cfg.chunk_target_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
            max_chunks=cfg.max_chunks_per_doc,
        )
        for sec in sections:
            for chunk in sec.chunks:
                all_chunk_meta.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": doc_id,
                    "section_id": chunk.section_id,
                    "section_path": chunk.section_path,
                    "domain": source_type,
                    "source_type": source_type,
                    "raw_text": chunk.raw_text,
                })
    logger.info(f"Segmented to {len(all_chunk_meta)} chunks across {len(docs)} docs")

    # 2. Run separate-call mode
    t0 = time.time()
    sep_distilled: list = []
    sep_cognitive: list = []
    for cm in all_chunk_meta:
        d, c = _per_chunk_extract_separate(cfg, cm)
        sep_distilled.extend(d)
        sep_cognitive.extend(c)
    sep_seconds = time.time() - t0
    logger.info(
        f"Separate-call mode: {len(sep_distilled)} distilled + {len(sep_cognitive)} cognitive in {sep_seconds:.1f}s"
    )

    # 3. Run combined-call mode
    t0 = time.time()
    cmb_distilled: list = []
    cmb_cognitive: list = []
    for cm in all_chunk_meta:
        d, c = _per_chunk_extract_combined(cfg, cm)
        cmb_distilled.extend(d)
        cmb_cognitive.extend(c)
    cmb_seconds = time.time() - t0
    logger.info(
        f"Combined-call mode: {len(cmb_distilled)} distilled + {len(cmb_cognitive)} cognitive in {cmb_seconds:.1f}s"
    )

    # 4. Per-type stats
    def _ctype_dist(entries: list) -> Dict[str, int]:
        cnt: Dict[str, int] = collections.Counter()
        for e in entries:
            cnt[e.memory_type] += 1
        return dict(cnt)

    sep_ct = _ctype_dist(sep_cognitive)
    cmb_ct = _ctype_dist(cmb_cognitive)
    sep_dt = _ctype_dist(sep_distilled)
    cmb_dt = _ctype_dist(cmb_distilled)

    # 5. Verdict
    def _safe_ratio(a, b):
        return float(a) / float(b) if b else (1.0 if a == 0 else float("inf"))

    distilled_count_ratio = _safe_ratio(len(cmb_distilled), len(sep_distilled))
    cognitive_count_ratio = _safe_ratio(len(cmb_cognitive), len(sep_cognitive))

    sep_types_set = set(sep_ct.keys())
    cmb_types_set = set(cmb_ct.keys())
    cognitive_type_coverage_loss = len(sep_types_set - cmb_types_set)

    per_type_max_drop = 0.0
    per_type_drops: Dict[str, float] = {}
    for t, sep_n in sep_ct.items():
        cmb_n = cmb_ct.get(t, 0)
        drop = 1.0 - _safe_ratio(cmb_n, sep_n) if sep_n else 0.0
        per_type_drops[t] = drop
        if drop > per_type_max_drop:
            per_type_max_drop = drop

    verdict_pass = (
        distilled_count_ratio >= 0.80
        and cognitive_count_ratio >= 0.80
        and cognitive_type_coverage_loss <= 2
        and per_type_max_drop <= 0.50
    )

    result = {
        "n_docs": len(docs),
        "n_chunks": len(all_chunk_meta),
        "separate": {
            "distilled_count": len(sep_distilled),
            "cognitive_count": len(sep_cognitive),
            "distilled_type_dist": sep_dt,
            "cognitive_type_dist": sep_ct,
            "wall_seconds": sep_seconds,
        },
        "combined": {
            "distilled_count": len(cmb_distilled),
            "cognitive_count": len(cmb_cognitive),
            "distilled_type_dist": cmb_dt,
            "cognitive_type_dist": cmb_ct,
            "wall_seconds": cmb_seconds,
        },
        "metrics": {
            "distilled_count_ratio": round(distilled_count_ratio, 4),
            "cognitive_count_ratio": round(cognitive_count_ratio, 4),
            "cognitive_type_coverage_loss": cognitive_type_coverage_loss,
            "per_type_max_drop": round(per_type_max_drop, 4),
            "per_type_drops": {k: round(v, 4) for k, v in per_type_drops.items()},
            "wall_speedup": round(_safe_ratio(sep_seconds, cmb_seconds), 2),
        },
        "thresholds": {
            "distilled_count_ratio_min": 0.80,
            "cognitive_count_ratio_min": 0.80,
            "cognitive_type_coverage_loss_max": 2,
            "per_type_max_drop_max": 0.50,
        },
        "verdict": "PASS" if verdict_pass else "FAIL",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-parquet", required=True)
    parser.add_argument("--tier-manifest", default=None)
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chat-model", default=os.getenv("LLM_CHAT_MODEL", "YOUR_CHAT_MODEL"))
    parser.add_argument("--llm-api-base", default=os.getenv("LLM_API_BASE", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import pandas as pd
    from megamem.document_eval.types import DocumentRetrievalConfig

    docs_df = pd.read_parquet(args.docs_parquet)
    if args.tier_manifest:
        tier = pd.read_parquet(args.tier_manifest)
        keep = set(tier["doc_id"].tolist())
        docs_df = docs_df[docs_df["doc_id"].isin(keep)].copy()
    # Stable subsampling by seed
    docs_df = docs_df.sample(n=min(args.n_docs, len(docs_df)), random_state=args.seed).reset_index(drop=True)
    docs = [
        {
            "doc_id": str(r["doc_id"]),
            "title": str(r.get("title") or ""),
            "source_type": str(r.get("source_type") or ""),
            "content": str(r.get("content") or ""),
        }
        for _, r in docs_df.iterrows()
    ]

    cfg = DocumentRetrievalConfig(
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        chat_model_id=args.chat_model,
        judge_model_id=os.getenv("LLM_JUDGE_MODEL", "YOUR_JUDGE_MODEL"),
        chroma_path="/tmp/option_e_preflight_dummy_chroma",
        collection_prefix="option_e_preflight",
        use_local_embedding=True,
        chunk_target_tokens=400,
        distilled_memory_per_chunk_budget=3,
        use_combined_distilled_cognitive_prompt=False,  # we toggle manually inside
    )

    result = run_preflight(docs, cfg, n_docs=args.n_docs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
