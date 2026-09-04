"""Document build pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from megamem.document_eval.chunking import count_tokens, segment_document
from megamem.document_eval.extractors import (
    extract_cognitive_relations,
    extract_combined_distilled_and_cognitive,
    extract_distilled_memories,
    summarize_document,
    summarize_section,
)
from megamem.document_eval.storage import DocumentStorage
from megamem.document_eval.types import (
    CognitiveEntry,
    DistilledMemoryEntry,
    DocumentNode,
    DocumentRetrievalConfig,
    RawChunkEntry,
    SectionNode,
)

logger = logging.getLogger(__name__)


PROGRESS_FILENAME = "_doc_build_progress.jsonl"


class DocumentBuildPipeline:
    """Build raw_chunks + distilled_memory + cognitive + summaries in one pass.

    Build phases can be toggled to support per-algorithm ablations:
    - build_distilled=False           → DDI Stream B disabled (cheap baselines)
    - build_cognitive=False           → CDM build disabled
    - build_section_summaries=False   → HDM Layer 3 disabled
    - build_document_summaries=False  → HDM Layer 2 disabled

    The raw_chunks collection is always built (everything else depends on it).
    """

    def __init__(self, cfg: DocumentRetrievalConfig, storage: Optional[DocumentStorage] = None):
        self.cfg = cfg
        self.storage = storage or DocumentStorage(cfg)
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats: Dict[str, Any] = {
            "n_documents": 0,
            "n_documents_resumed_skipped": 0,
            "n_sections": 0,
            "n_raw_chunks": 0,
            "n_distilled_memories": 0,
            "n_cognitive_entries": 0,
            "n_doc_summaries": 0,
            "n_section_summaries": 0,
            "tokens_raw": 0,
            "extract_seconds": 0.0,
            "build_seconds": 0.0,
            "skipped_docs": 0,
            "n_input_dedup_removed": 0,
            "n_shards_completed": 0,
        }

    @property
    def _progress_path(self) -> str:
        return os.path.join(self.cfg.chroma_path, PROGRESS_FILENAME)

    def _load_progress(self) -> Set[str]:
        """Return the set of doc_ids previously marked done in this chroma_path."""
        p = self._progress_path
        if not os.path.exists(p):
            return set()
        done: Set[str] = set()
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("stage") == "done" and rec.get("doc_id"):
                            done.add(rec["doc_id"])
                    except json.JSONDecodeError:
                        # Tolerate a truncated final line from a crash
                        continue
        except OSError as exc:
            logger.warning(f"Failed to read progress journal {p}: {exc}")
        return done

    def _append_progress(self, recs: List[Dict[str, Any]]) -> None:
        """Append one or more progress records and fsync (for crash safety)."""
        if not recs:
            return
        p = self._progress_path
        os.makedirs(self.cfg.chroma_path, exist_ok=True)
        # Open with line buffering + fsync to ensure durability after each shard.
        with open(p, "a", buffering=1) as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # best effort; not all filesystems support fsync on regular files

    def reset_progress(self) -> None:
        """Drop the progress journal (caller's responsibility before force_rebuild)."""
        p = self._progress_path
        if os.path.exists(p):
            os.remove(p)

    def build(
        self,
        documents: Iterable[Dict[str, Any]],
        *,
        build_raw_chunks: bool = True,
        build_distilled: bool = True,
        build_cognitive: bool = True,
        build_section_summaries: bool = True,
        build_document_summaries: bool = True,
        max_extract_workers: int = 4,
        force_rebuild: bool = True,
        resume: bool = False,
        shard_size: int = 25,
        progress_every: int = 25,
    ) -> Dict[str, Any]:
        """Run the build pipeline."""
        t_start = time.time()
        self._reset_stats()

        if force_rebuild:
            logger.info("force_rebuild=True: dropping all doc_eval collections + progress journal")
            self.storage.reset_all()
            self.reset_progress()
            resume = False  # any meaningful resume requires the journal we just wiped

        # Stage 1: dedup input by doc_id and apply resume filter
        docs = list(documents)
        seen_ids: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        n_input = len(docs)
        for d in docs:
            did = d.get("doc_id", "")
            if did and did not in seen_ids:
                seen_ids.add(did)
                deduped.append(d)
        n_dedup_dropped = n_input - len(deduped)
        self.stats["n_input_dedup_removed"] = n_dedup_dropped
        if n_dedup_dropped:
            logger.warning(f"deduplicated {n_dedup_dropped} duplicate doc_id rows from input")

        if resume:
            already_done = self._load_progress()
            if already_done:
                before = len(deduped)
                deduped = [d for d in deduped if d.get("doc_id") not in already_done]
                self.stats["n_documents_resumed_skipped"] = before - len(deduped)
                logger.info(
                    f"resume=True: progress journal at {self._progress_path!s} marks "
                    f"{len(already_done)} doc(s) as done; {self.stats['n_documents_resumed_skipped']} "
                    f"of input present in journal; processing remaining {len(deduped)} doc(s)"
                )

        docs = deduped
        self.stats["n_documents"] = len(docs)
        logger.info(f"Pipeline build start: {len(docs)} doc(s) to process; shard_size={shard_size}")

        # Stage 2: process docs in shards
        shards = [docs[i : i + shard_size] for i in range(0, len(docs), shard_size)]
        for shard_idx, shard_docs in enumerate(shards):
            self._process_shard(
                shard_docs,
                shard_idx=shard_idx,
                n_shards=len(shards),
                build_raw_chunks=build_raw_chunks,
                build_distilled=build_distilled,
                build_cognitive=build_cognitive,
                build_section_summaries=build_section_summaries,
                build_document_summaries=build_document_summaries,
                max_extract_workers=max_extract_workers,
                progress_every=progress_every,
            )
            self.stats["n_shards_completed"] += 1

        self.stats["build_seconds"] = time.time() - t_start
        logger.info(f"Pipeline build complete in {self.stats['build_seconds']:.1f}s; stats={self.stats}")
        return self.stats

    def _process_shard(
        self,
        shard_docs: List[Dict[str, Any]],
        *,
        shard_idx: int,
        n_shards: int,
        build_raw_chunks: bool,
        build_distilled: bool,
        build_cognitive: bool,
        build_section_summaries: bool,
        build_document_summaries: bool,
        max_extract_workers: int,
        progress_every: int,
    ) -> None:
        """Process one shard of docs atomically.

        Order: segment → extract distilled+cognitive (parallel) → section/doc
        summaries → batched upsert per collection → append progress journal.
        """
        t_shard = time.time()
        shard_label = f"shard {shard_idx + 1}/{n_shards} ({len(shard_docs)} docs)"
        logger.info(f"[{shard_label}] start")

        # 1. Segment all docs in shard
        shard_raw: List[RawChunkEntry] = []
        shard_section_specs: List[Dict[str, Any]] = []
        shard_doc_meta: List[Dict[str, Any]] = []
        docs_with_content: List[Dict[str, Any]] = []

        for doc in shard_docs:
            doc_id = doc["doc_id"]
            title = (doc.get("title") or "")[:300]
            source_type = doc.get("source_type", "")
            content = doc.get("content") or ""
            if not content.strip():
                self.stats["skipped_docs"] += 1
                # Mark as done so resume skips empty docs (no work to retry).
                self._append_progress([{
                    "doc_id": doc_id,
                    "stage": "done",
                    "ts": time.time(),
                    "shard_idx": shard_idx,
                    "skipped_empty": True,
                }])
                continue
            sections = segment_document(
                doc_id,
                content,
                target_tokens=self.cfg.chunk_target_tokens,
                overlap_tokens=self.cfg.chunk_overlap_tokens,
                max_chunks=self.cfg.max_chunks_per_doc,
            )
            self.stats["n_sections"] += len(sections)
            for sec in sections:
                for chunk in sec.chunks:
                    e = RawChunkEntry(
                        chunk_id=chunk.chunk_id,
                        document_id=doc_id,
                        section_id=chunk.section_id,
                        section_path=chunk.section_path,
                        position=chunk.position,
                        raw_text=chunk.raw_text,
                        domain=source_type,
                        source_type=source_type,
                        title=title,
                        token_count=chunk.token_count,
                    )
                    shard_raw.append(e)
                    self.stats["tokens_raw"] += chunk.token_count
                shard_section_specs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "source_type": source_type,
                    "section_id": sec.section_id,
                    "section_path": sec.section_path,
                    "section_title": sec.title,
                    "level": sec.level,
                    "chunk_ids": [c.chunk_id for c in sec.chunks],
                    "body": "\n\n".join(c.raw_text for c in sec.chunks)[:8000],
                    "token_count": sum(c.token_count for c in sec.chunks),
                    "prev_section_id": sec.prev_section_id,
                    "next_section_id": sec.next_section_id,
                })
            shard_doc_meta.append({
                "doc_id": doc_id,
                "title": title,
                "source_type": source_type,
                "section_ids": [s.section_id for s in sections],
                "section_count": len(sections),
                "token_count": sum(c.token_count for s in sections for c in s.chunks),
            })
            docs_with_content.append(doc)

        if not docs_with_content:
            logger.info(f"[{shard_label}] all docs empty; nothing to upsert")
            return

        # 2. Extract distilled + cognitive (parallel across chunks in shard)
        shard_distilled: List[DistilledMemoryEntry] = []
        shard_cognitive: List[CognitiveEntry] = []
        if (build_distilled or build_cognitive) and shard_raw:
            t_extract = time.time()
            self._extract_chunks_to_lists(
                shard_raw,
                shard_distilled,
                shard_cognitive,
                build_distilled=build_distilled,
                build_cognitive=build_cognitive,
                max_workers=max_extract_workers,
                progress_every=progress_every,
            )
            self.stats["extract_seconds"] += time.time() - t_extract

        # 3. Section summaries
        shard_section_nodes: List[SectionNode] = []
        if build_section_summaries:
            for s in shard_section_specs:
                summary = summarize_section(
                    self.cfg,
                    title=s["section_title"],
                    body=s["body"],
                    min_tokens_for_llm=self.cfg.section_summary_min_tokens,
                )
                shard_section_nodes.append(SectionNode(
                    section_id=s["section_id"],
                    document_id=s["doc_id"],
                    section_path=s["section_path"],
                    section_title=s["section_title"],
                    level=s["level"],
                    summary=summary,
                    chunk_ids=s["chunk_ids"],
                    prev_section_id=s["prev_section_id"],
                    next_section_id=s["next_section_id"],
                    domain=s["source_type"],
                    source_type=s["source_type"],
                    token_count=s["token_count"],
                ))

        # 4. Doc summaries (depends on section summaries; we can compose in-memory
        #    from shard_section_nodes built above; no chroma round-trip needed.)
        shard_doc_nodes: List[DocumentNode] = []
        if build_document_summaries:
            # Build a per-doc map of section summaries from what we just computed.
            section_summary_text_by_id: Dict[str, str] = {
                n.section_id: f"{n.section_path}: {n.summary}" for n in shard_section_nodes
            }
            for d in shard_doc_meta:
                section_summaries = [
                    section_summary_text_by_id.get(s, "") for s in d["section_ids"]
                ]
                section_summaries = [s for s in section_summaries if s]
                doc_summary = summarize_document(self.cfg, d["title"], section_summaries)
                shard_doc_nodes.append(DocumentNode(
                    document_id=d["doc_id"],
                    title=d["title"],
                    doc_type=d["source_type"],
                    domain=d["source_type"],
                    source_type=d["source_type"],
                    section_ids=d["section_ids"],
                    summary=doc_summary,
                    token_count=d["token_count"],
                ))

        # 5. Batched upsert per collection (one trip per collection per shard)
        if build_raw_chunks and shard_raw:
            self._upsert_raw_chunks(shard_raw)
        if build_distilled and shard_distilled:
            self._upsert_distilled(shard_distilled)
        if build_cognitive and shard_cognitive:
            self._upsert_cognitive(shard_cognitive)
        if build_section_summaries and shard_section_nodes:
            self._upsert_section_summaries(shard_section_nodes)
        if build_document_summaries and shard_doc_nodes:
            self._upsert_document_summaries(shard_doc_nodes)

        now = time.time()
        recs = [{
            "doc_id": d["doc_id"],
            "stage": "done",
            "ts": now,
            "shard_idx": shard_idx,
            "n_chunks": sum(1 for r in shard_raw if r.document_id == d["doc_id"]),
        } for d in docs_with_content]
        self._append_progress(recs)

        self.stats["n_raw_chunks"] += len(shard_raw)
        self.stats["n_distilled_memories"] += len(shard_distilled)
        self.stats["n_cognitive_entries"] += len(shard_cognitive)
        self.stats["n_section_summaries"] += len(shard_section_nodes)
        self.stats["n_doc_summaries"] += len(shard_doc_nodes)

        logger.info(
            f"[{shard_label}] done in {time.time() - t_shard:.1f}s "
            f"(+{len(shard_raw)} chunks, +{len(shard_distilled)} distilled, "
            f"+{len(shard_cognitive)} cognitive, +{len(shard_section_nodes)} secs, "
            f"+{len(shard_doc_nodes)} docs)"
        )

    def _extract_chunks_to_lists(
        self,
        raws: List[RawChunkEntry],
        out_distilled: List[DistilledMemoryEntry],
        out_cognitive: List[CognitiveEntry],
        *,
        build_distilled: bool,
        build_cognitive: bool,
        max_workers: int,
        progress_every: int,
    ) -> None:
        use_combined = bool(
            getattr(self.cfg, "use_combined_distilled_cognitive_prompt", False)
            and build_distilled
            and build_cognitive
        )

        def _process(r: RawChunkEntry) -> tuple:
            d_local: List[DistilledMemoryEntry] = []
            c_local: List[CognitiveEntry] = []
            if use_combined:
                # Stage 2 Option E: 1 LLM call returns both lists.
                d_local, c_local = extract_combined_distilled_and_cognitive(
                    self.cfg,
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    section_id=r.section_id,
                    section_path=r.section_path,
                    domain=r.domain,
                    source_type=r.source_type,
                    raw_text=r.raw_text,
                )
                return d_local, c_local
            if build_distilled:
                d_local = extract_distilled_memories(
                    self.cfg,
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    section_id=r.section_id,
                    section_path=r.section_path,
                    domain=r.domain,
                    source_type=r.source_type,
                    raw_text=r.raw_text,
                )
            if build_cognitive:
                c_local = extract_cognitive_relations(
                    self.cfg,
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    section_id=r.section_id,
                    section_path=r.section_path,
                    domain=r.domain,
                    source_type=r.source_type,
                    raw_text=r.raw_text,
                )
            return d_local, c_local

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process, r) for r in raws]
            for fut in as_completed(futures):
                d_local, c_local = fut.result()
                out_distilled.extend(d_local)
                out_cognitive.extend(c_local)
                done += 1
                if done % progress_every == 0:
                    logger.info(
                        f"    extract: {done}/{len(raws)} chunks in shard "
                        f"(+{len(out_distilled)} distilled, +{len(out_cognitive)} cognitive total in shard)"
                    )

    def _upsert_raw_chunks(self, raws: List[RawChunkEntry]) -> None:
        ids = [r.chunk_id for r in raws]
        docs = [r.raw_text for r in raws]
        meta = [r.to_chroma_metadata() for r in raws]
        self.storage.upsert("raw_chunks", ids=ids, documents=docs, metadatas=meta)

    def _upsert_distilled(self, distilled: List[DistilledMemoryEntry]) -> None:
        ids = [d.memory_id for d in distilled]
        docs = [f"{d.index}: {d.value}" for d in distilled]
        meta = [d.to_chroma_metadata() for d in distilled]
        self.storage.upsert("distilled_memory", ids=ids, documents=docs, metadatas=meta)

    def _upsert_cognitive(self, cognitive: List[CognitiveEntry]) -> None:
        ids = [c.cognitive_id for c in cognitive]
        docs = [f"{c.memory_type}: {c.index}. {c.value}" for c in cognitive]
        meta = [c.to_chroma_metadata() for c in cognitive]
        self.storage.upsert("cognitive", ids=ids, documents=docs, metadatas=meta)

    def _upsert_section_summaries(self, nodes: List[SectionNode]) -> None:
        ids = [n.section_id for n in nodes]
        docs = [f"{n.section_path}: {n.summary}" for n in nodes]
        meta = [n.to_chroma_metadata() for n in nodes]
        self.storage.upsert("section_summaries", ids=ids, documents=docs, metadatas=meta)

    def _upsert_document_summaries(self, nodes: List[DocumentNode]) -> None:
        ids = [n.document_id for n in nodes]
        docs = [f"{n.title}: {n.summary}" for n in nodes]
        meta = [n.to_chroma_metadata() for n in nodes]
        self.storage.upsert("doc_summaries", ids=ids, documents=docs, metadatas=meta)
