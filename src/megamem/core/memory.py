import enum
import logging
import time
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig
from chromadb.api.types import Where

from megamem.core.base import MemoryBase
from megamem.core.local_memory_store import LocalMemoryStore
from megamem.core.memory_entry import MemoryEntry
from megamem.core.memory_filter import MemoryFilter
from megamem.core.query_generator import QueryGenerator

from megamem.utils.llm import ChatCompletionModel
from megamem.utils.log import log_memory_operation
from megamem.utils.memory import combine_list, merge_with_rrf
from megamem.utils.misc import context_to_str, extract_user_id_from_where, index_to_id

logger = logging.getLogger(__name__)


class QueryMode(enum.Enum):
    ORIGINAL = 1
    PRIMARY_ONLY = 2
    CUE_ONLY = 3
    BOTH = 4


class AgentMemory(MemoryBase):

    def __init__(self, cfg: DictConfig, user_id: str):
        self.cfg = cfg

        self.user_id = user_id

        self.MAX_CONTEXT_TOKENS = 128000

        self.query_generator = QueryGenerator(cfg)
        self.memory_filter = MemoryFilter(cfg)

        self.multimodal_support = cfg.memory.get("multimodal_support", True)

        self._llm_client = ChatCompletionModel(cfg)

        self._store = LocalMemoryStore(cfg, user_id)

        self.QUERY_SCORE_THRESHOLD = (
            cfg.memory.query_score_threshold
        )

    def get_user_id(self) -> str:
        return self.user_id

    def _query_result(
        self,
        queries: List[str],
        top_k: int,
        where: Optional[Where] = None,
        query_mode: QueryMode = QueryMode.ORIGINAL,
        include: Optional[List[str]] = None,
        return_history: bool = False,
    ):
        memory_results = []
        extracted: set = set()

        factual_memory_condition = {"memory_type": {"$eq": "factual"}}

        if query_mode == QueryMode.CUE_ONLY:
            where = {"linked_memory": {"$ne": ""}}
        elif query_mode == QueryMode.PRIMARY_ONLY:
            query_condition = {"linked_memory": {"$eq": ""}}
            where = {"$and": [query_condition, factual_memory_condition]}

        for query in queries:
            results: List[MemoryEntry] = self._store.query(query, top_k, where, include)

            scored_pairs = [(entry, entry.score) for entry in results]
            scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

            for entry, score in scored_pairs:

                if score < self.QUERY_SCORE_THRESHOLD:
                    break

                if entry.is_cue_index():
                    for primary_index in entry.get_linked_memories():
                        primary_entry = self._store.get(primary_index)

                        if not primary_entry:
                            logger.warning(f"Primary memory entry cannot found: {primary_index}")
                            continue

                        value = primary_entry.get_memory_value()
                        if value in extracted:
                            continue

                        memory_results.append(primary_entry)
                        extracted.add(value)
                else:
                    value = entry.get_memory_value()

                    if value in extracted:
                        continue

                    memory_results.append(entry)
                    extracted.add(value)

        return memory_results[:top_k]

    def _perform_hybrid_search(
        self,
        context: str,
        where: Optional[Where] = None,
    ) -> List[MemoryEntry]:
        hybrid_method = self.cfg.memory.get("hybrid_search_method", "bm25")
        hybrid_top_k = self.cfg.memory.get("hybrid_top_k", 10)

        if hybrid_method == "bm25":
            target_user_id = extract_user_id_from_where(where) or self.user_id

            if target_user_id and target_user_id not in self._store._bm25_indices:
                logger.info(f"Building BM25 index for user {target_user_id} before first search")
                self._store.build_bm25_index(user_id=target_user_id)

        hybrid_results: List[MemoryEntry] = []

        if hybrid_method == "bm25":
            bm25_threshold = self.cfg.memory.get("bm25_score_threshold", 0.4)
            hybrid_results = self._store.bm25_search(context, hybrid_top_k, where, bm25_threshold)

        elif hybrid_method == "keyword":
            keywords = self.query_generator.extract_keywords(context)
            if keywords:
                hybrid_results = self._store.keyword_search(keywords, hybrid_top_k, where)

        else:
            raise ValueError(f"Unsupported hybrid search method: {hybrid_method}")

        return hybrid_results

    def _merge_results_with_rrf(
        self,
        result_lists: List[List[MemoryEntry]],
        weights: Optional[List[float]] = None,
        k: int = 60,
    ) -> List[MemoryEntry]:
        return merge_with_rrf(result_lists, weights=weights, k=k)

    def _search_source_cues(
        self,
        step,
        query_text: str,
        top_k: int,
        latency_tracker,
    ) -> List[MemoryEntry]:
        from megamem.core.retrieval_planner import (
            build_where_clause, get_string_filters,
            merge_where_clauses, apply_string_filters,
        )

        step_where = build_where_clause(step)

        sc_condition = {"cue_type": {"$eq": "source"}}
        effective_where = merge_where_clauses(step_where, sc_condition)

        if latency_tracker:
            with latency_tracker.track("search_source_cues"):
                raw_results = self._store.query(
                    query_text, top_k * 3, effective_where
                )
        else:
            raw_results = self._store.query(
                query_text, top_k * 3, effective_where
            )

        results = [
            entry for entry in raw_results
            if entry.score >= self.QUERY_SCORE_THRESHOLD
        ]

        string_filters = get_string_filters(step)
        if string_filters:
            results = apply_string_filters(results, string_filters)

        logger.info(
            f"SS(source_cues) {step.step_id}: "
            f"{len(raw_results)} raw → {len(results)} after threshold+filters"
        )

        return results

    def _search_primary_memories(
        self,
        step,
        query_text: str,
        top_k: int,
        filtered_source_cues,
        latency_tracker,
    ) -> List[MemoryEntry]:
        primary_condition = {"$and": [
            {"linked_memory": {"$eq": ""}},
            {"memory_type": {"$eq": "factual"}},
        ]}

        if step.scope == "filtered_results" and filtered_source_cues:
            linked_ids: set = set()
            for cue in filtered_source_cues:
                linked_ids.update(cue.get_linked_memories())

            if not linked_ids:
                logger.warning(
                    f"SS(primary_memories) {step.step_id}: "
                    f"no linked primary IDs from source cues."
                )
                return []

            search_n = max(top_k * 5, 50)

            if latency_tracker:
                with latency_tracker.track("search_primary_scoped"):
                    raw_results = self._store.query(
                        query_text, search_n, primary_condition
                    )
            else:
                raw_results = self._store.query(
                    query_text, search_n, primary_condition
                )

            results = [
                entry for entry in raw_results
                if entry.score >= self.QUERY_SCORE_THRESHOLD
                and entry.index in linked_ids
            ]

            logger.info(
                f"SS(primary_memories, scoped) {step.step_id}: "
                f"{len(raw_results)} raw → {len(results)} after "
                f"threshold+scope ({len(linked_ids)} linked IDs)"
            )
            return results[:top_k]

        if latency_tracker:
            with latency_tracker.track("search_primary"):
                raw_results = self._store.query(
                    query_text, top_k, primary_condition
                )
        else:
            raw_results = self._store.query(
                query_text, top_k, primary_condition
            )

        results = [
            entry for entry in raw_results
            if entry.score >= self.QUERY_SCORE_THRESHOLD
        ]

        logger.info(
            f"SS(primary_memories, all) {step.step_id}: "
            f"{len(raw_results)} raw → {len(results)} after threshold"
        )
        return results

    _VALID_PLAN_SHAPES = {
        (("FILTER", None), ("RESOLVE", None)),

        (("SEMANTIC_SEARCH", "source_cues"), ("RESOLVE", None)),
        (("SEMANTIC_SEARCH", "source_cues"), ("SEMANTIC_SEARCH", "primary_memories")),
        (("SEMANTIC_SEARCH", "primary_memories"),),
    }

    _VALID_OPS = {"FILTER", "SEMANTIC_SEARCH", "RESOLVE"}
    _VALID_TARGETS = {"source_cues", "primary_memories"}
    _VALID_RETURN_MODES = {"metadata_summary", "full_content"}
    _VALID_SCOPES = {"all_sources", "filtered_results"}
    _VALID_DATA_TYPES = {"mail", "doc", "teams"}

    def _validate_plan(self, plan, context: str) -> List[str]:
        issues: List[str] = []
        steps = plan.steps

        if len(steps) == 0:
            issues.append("ERROR: plan has 0 steps")
            return issues
        if len(steps) > 2:
            issues.append(
                f"ERROR: plan has {len(steps)} steps (max 2). "
                f"Ops: {[s.op for s in steps]}"
            )
            return issues

        for step in steps:
            sid = step.step_id

            if step.op not in self._VALID_OPS:
                issues.append(f"ERROR: {sid} has unknown op '{step.op}'")
                continue

            if step.op == "SEMANTIC_SEARCH":
                if step.target and step.target not in self._VALID_TARGETS:
                    issues.append(
                        f"ERROR: {sid} SS target '{step.target}' "
                        f"not in {self._VALID_TARGETS}"
                    )
                if step.target == "primary_memories":
                    for fld in ("sender", "recipients", "author", "title", "data_type", "participants", "topic", "conversation_type"):
                        if getattr(step, fld, None):
                            issues.append(
                                f"WARN: {sid} SS(primary_memories) has "
                                f"{fld}='{getattr(step, fld)}' which is ignored "
                                f"(primary memories don't carry source metadata)"
                            )

            if step.data_type and step.data_type not in self._VALID_DATA_TYPES:
                issues.append(
                    f"WARN: {sid} has unknown data_type '{step.data_type}', "
                    f"stripping data_type and associated metadata fields"
                )
                step.data_type = None
                for fld in ("sender", "recipients", "author", "title", "participants", "topic", "conversation_type"):
                    setattr(step, fld, None)

            if step.op == "RESOLVE":
                if step.return_mode and step.return_mode not in self._VALID_RETURN_MODES:
                    issues.append(
                        f"ERROR: {sid} RESOLVE return_mode "
                        f"'{step.return_mode}' not in {self._VALID_RETURN_MODES}"
                    )

            if step.scope and step.scope not in self._VALID_SCOPES:
                issues.append(
                    f"WARN: {sid} unknown scope '{step.scope}', "
                    f"will fall through to all_sources"
                )

        shape = tuple(
            (s.op, s.target if s.op == "SEMANTIC_SEARCH" else None)
            for s in steps
        )
        if shape not in self._VALID_PLAN_SHAPES:
            issues.append(
                f"ERROR: unrecognized plan shape {shape}. "
                f"Expected one of: {self._VALID_PLAN_SHAPES}"
            )

        if steps[0].op == "RESOLVE":
            issues.append(
                "ERROR: RESOLVE cannot be the first step "
                "(no source cues from a preceding FILTER or SS)"
            )

        return issues

    def _execute_planner_query(
        self,
        context: str,
        top_k: int = 5,
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        from megamem.core.retrieval_planner import (
            RetrievalPlanner, build_where_clause,
            get_string_filters, apply_string_filters,
            resolve_source_cues, build_source_cue_filter,
        )

        planner = RetrievalPlanner(self.cfg, self._llm_client)

        if latency_tracker:
            with latency_tracker.track("planner"):
                plan = planner.plan(context)
        else:
            plan = planner.plan(context)

        logger.info(
            f"Planner produced {len(plan.steps)} steps for: '{context[:50]}...' "
            f"| reasoning: {plan.reasoning[:80]}"
        )

        self._last_plan = plan
        self._last_source_cues = None

        issues = self._validate_plan(plan, context)
        for issue in issues:
            logger.warning(f"Plan validation [{context[:40]}...]: {issue}")
        if any(item.startswith("ERROR:") for item in issues):
            logger.warning(
                f"Plan validation failed for '{context[:50]}...'. "
                f"Falling back to SS(primary_memories). Issues: {issues}"
            )
            from megamem.core.retrieval_planner import RetrievalStep
            fallback_step = RetrievalStep(
                step_id="S1_fallback",
                op="SEMANTIC_SEARCH",
                target="primary_memories",
                scope="all_sources",
                query_text=context,
            )
            return self._search_primary_memories(
                fallback_step, context, top_k, None, latency_tracker
            )

        filtered_source_cues = None
        memory_results: List[MemoryEntry] = []

        for step in plan.steps:

            if step.op == "FILTER":
                where_clause = build_where_clause(step)
                string_filters = get_string_filters(step)

                filter_where = build_source_cue_filter(where_clause)
                filtered_source_cues = self._store.filter(
                    where=filter_where, limit=top_k * 5
                )
                if string_filters:
                    filtered_source_cues = apply_string_filters(
                        filtered_source_cues, string_filters
                    )

                logger.info(
                    f"FILTER {step.step_id}: where={where_clause}, "
                    f"string_filters={string_filters}, "
                    f"found {len(filtered_source_cues)} source cues"
                )

            elif step.op == "SEMANTIC_SEARCH":
                query_text = step.query_text or context

                if step.target == "source_cues":

                    filtered_source_cues = self._search_source_cues(
                        step, query_text, top_k, latency_tracker,
                    )
                    self._last_source_cues = filtered_source_cues
                    memory_results = filtered_source_cues

                elif step.target == "primary_memories":
                    memory_results = self._search_primary_memories(
                        step, query_text, top_k,
                        filtered_source_cues, latency_tracker,
                    )

                else:
                    logger.warning(
                        f"SS {step.step_id}: unknown target '{step.target}', "
                        f"defaulting to primary_memories"
                    )
                    memory_results = self._search_primary_memories(
                        step, query_text, top_k,
                        filtered_source_cues, latency_tracker,
                    )

            elif step.op == "RESOLVE":
                if filtered_source_cues is None:
                    logger.warning(
                        f"RESOLVE {step.step_id}: no source cues "
                        f"from previous step. Returning empty results."
                    )
                    memory_results = []
                else:
                    memory_results = resolve_source_cues(
                        filtered_source_cues,
                        return_mode=step.return_mode or "metadata_summary",
                        metadata_fields=step.metadata_fields,
                    )

            else:
                logger.warning(
                    f"Unknown op '{step.op}' in {step.step_id}, skipping"
                )

        return memory_results[:top_k]

    def planner_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        context = context_to_str(context)
        return self._execute_planner_query(
            context=context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Where] = None,
        query_mode: QueryMode = QueryMode.ORIGINAL,
        include: Optional[List[str]] = None,
        enhance_query: bool = True,
        return_history: bool = False,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        latency_tracker = None,
    ):

        context = context_to_str(context)

        memory_results = []

        query_start_time = time.time()

        if enhance_query:
            query_gen_start = time.time()
            queries = self.query_generator.generate_queries(context)
            query_gen_time = time.time() - query_gen_start
            logger.info(f"[LATENCY] Query generation took {query_gen_time:.3f}s")
        else:
            queries = [context]

        primary_results = []
        cue_results = []
        hybrid_results = []

        if query_mode == QueryMode.ORIGINAL:
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.ORIGINAL, include, return_history
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.ORIGINAL, include, return_history
                )
            memory_results = primary_results
        elif query_mode == QueryMode.PRIMARY_ONLY:
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history
                )
            memory_results = primary_results
        elif query_mode == QueryMode.CUE_ONLY:
            if latency_tracker:
                with latency_tracker.track("search_cue"):
                    cue_results = self._query_result(
                        queries, self.cfg.memory.cue_top_k, where, QueryMode.CUE_ONLY, include, return_history
                    )
            else:
                cue_results = self._query_result(
                    queries, self.cfg.memory.cue_top_k, where, QueryMode.CUE_ONLY, include, return_history
                )
            memory_results = cue_results
        elif query_mode == QueryMode.BOTH:
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history
                    )
                with latency_tracker.track("search_cue"):
                    cue_results = self._query_result(
                        queries,
                        self.cfg.memory.cue_top_k,
                        where,
                        QueryMode.CUE_ONLY,
                        include,
                        return_history,
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history
                )
                cue_results = self._query_result(
                    queries,
                    self.cfg.memory.cue_top_k,
                    where,
                    QueryMode.CUE_ONLY,
                    include,
                    return_history,
                )

        if enable_hybrid_search:
            try:
                if latency_tracker:
                    with latency_tracker.track("search_hybrid"):
                        hybrid_results = self._perform_hybrid_search(context, where)
                else:
                    hybrid_results = self._perform_hybrid_search(context, where)

                result_lists = []
                weights = []

                if primary_results:
                    result_lists.append(primary_results)
                    weights.append(2.0)

                if cue_results:
                    result_lists.append(cue_results)
                    weights.append(1.0)

                if hybrid_results:
                    result_lists.append(hybrid_results)
                    weights.append(1.0)

                rrf_start = time.time()
                if len(result_lists) > 1:
                    if latency_tracker:
                        with latency_tracker.track("search_rrf_merge"):
                            memory_results = self._merge_results_with_rrf(result_lists, weights)
                    else:
                        memory_results = self._merge_results_with_rrf(result_lists, weights)
                elif len(result_lists) == 1:
                    memory_results = result_lists[0]
                else:
                    memory_results = []
                rrf_time = time.time() - rrf_start
                logger.info(f"[LATENCY] RRF merging took {rrf_time:.3f}s")

            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}. Falling back to semantic search only.")
                if query_mode == QueryMode.BOTH and primary_results and cue_results:
                    if latency_tracker:
                        with latency_tracker.track("search_rrf_merge"):
                            memory_results = self._merge_results_with_rrf(
                                [primary_results, cue_results],
                                [2.0, 1.0]
                            )
                    else:
                        memory_results = self._merge_results_with_rrf(
                            [primary_results, cue_results],
                            [2.0, 1.0]
                        )
                elif primary_results:
                    memory_results = primary_results
                elif cue_results:
                    memory_results = cue_results
        else:
            if query_mode == QueryMode.BOTH and primary_results and cue_results:
                if latency_tracker:
                    with latency_tracker.track("search_rrf_merge"):
                        memory_results = self._merge_results_with_rrf(
                            [primary_results, cue_results],
                            [2.0, 1.0]
                        )
                else:
                    memory_results = self._merge_results_with_rrf(
                        [primary_results, cue_results],
                        [2.0, 1.0]
                    )

        if enable_llm_filter and memory_results:
            if latency_tracker:
                with latency_tracker.track("search_llm_filter"):
                    memory_results = self.memory_filter.filter_memory(
                        query=context,
                        memory_results=memory_results,
                    )
            else:
                memory_results = self.memory_filter.filter_memory(
                    query=context,
                    memory_results=memory_results,
                )
            return memory_results

        return memory_results[:top_k]

    def expand_by_session(
        self,
        memory_results: List[MemoryEntry],
        max_per_session: int = 5,
    ) -> List[MemoryEntry]:
        existing_values = {m.get_memory_value() for m in memory_results}

        sessions_seen: set = set()
        for m in memory_results:
            meta = m.get_metadata() if hasattr(m, "get_metadata") else {}
            conv_idx = meta.get("source_conv_idx")
            session = meta.get("source_session")
            if conv_idx is not None and session is not None:
                sessions_seen.add((conv_idx, session))

        if not sessions_seen:
            return memory_results

        expanded: List[MemoryEntry] = []
        for conv_idx, session in sessions_seen:
            where = {
                "$and": [
                    {"source_conv_idx": {"$eq": conv_idx}},
                    {"source_session": {"$eq": session}},
                    {"linked_memory": {"$eq": ""}},
                    {"memory_type": {"$eq": "factual"}},
                ]
            }
            try:
                hits: List[MemoryEntry] = self._store.filter(
                    where=where,
                    limit=max_per_session + len(memory_results),
                )
                for h in hits:
                    val = h.get_memory_value()
                    if val in existing_values:
                        continue
                    existing_values.add(val)
                    expanded.append(h)
                    if len(expanded) >= max_per_session * len(sessions_seen):
                        break
            except Exception as e:
                logger.debug(f"Session expansion failed for conv={conv_idx} session={session}: {e}")

        if expanded:
            logger.info(
                f"Session expansion added {len(expanded)} memories from "
                f"{len(sessions_seen)} sessions"
            )

        return memory_results + expanded

    def get_episodic_memories_for_results(
        self, memory_results: List[MemoryEntry]
    ) -> Dict[str, MemoryEntry]:
        episodic_ids: set = set()
        for entry in memory_results:
            if entry.episodic_memory_ids:
                episodic_ids.update(entry.episodic_memory_ids)

        episodic_memories: Dict[str, MemoryEntry] = {}
        for episodic_id in episodic_ids:
            episodic_entry = self._store.get(episodic_id)
            if episodic_entry:
                episodic_memories[episodic_id] = episodic_entry
            else:
                logger.warning(f"Episodic memory not found: {episodic_id}")

        return episodic_memories

    def get_all_cues(self) -> List[MemoryEntry]:
        return self._store.get_all_cues()

    def get(self, key: str) -> MemoryEntry:
        return self._store.get(key)

    def add(self, entry: MemoryEntry):
        assert (
            entry.is_primary_index()
        ), "Only primary memory entries can be added directly."

        exist_entry = self._store.get(entry.index)

        if exist_entry is not None:
            if entry.memory_type == "episodic":
                original_index = entry.index
                counter = 2
                while self._store.get(f"{original_index} ({counter})") is not None:
                    counter += 1
                entry.index = f"{original_index} ({counter})"
                logger.info(
                    f"Episodic memory index already exists. "
                    f"Renamed '{original_index}' to '{entry.index}'"
                )
            else:
                raise AssertionError(f"Memory entry {entry.index} already exists.")

        log_memory_operation("Add", entry, user_id=self.user_id)

        self._store.upsert(
            index=entry.index, value=entry.value, metadata=entry.get_metadata()
        )

        for cue_index in entry.get_cue_indices():

            cue_entry = self._store.get(cue_index)
            if (cue_entry and cue_entry.is_primary_index()) or cue_index == entry.index:
                entry.delete_cue_index(cue_index)
                continue

            linked_memory = entry.index
            if cue_entry and cue_entry.is_cue_index():
                linked_memory = combine_list(linked_memory, cue_entry.linked_memory)

            self._store.upsert(
                index=cue_index,
                value="",
                metadata={
                    "linked_memory": linked_memory,
                    "cue_type": "topical",
                },
            )

        for cue_index in entry.get_predictive_cue_indices():
            cue_entry = self._store.get(cue_index)
            if (cue_entry and cue_entry.is_primary_index()) or cue_index == entry.index:
                continue

            linked_memory = entry.index
            if cue_entry and cue_entry.is_cue_index():
                linked_memory = combine_list(linked_memory, cue_entry.linked_memory)

            self._store.upsert(
                index=cue_index,
                value="",
                metadata={
                    "linked_memory": linked_memory,
                    "cue_type": "predictive",
                },
            )

    def add_source_cue(
        self,
        source_description: str,
        linked_memory_indices: List[str],
        data_type: str = "",
        timestamp_unix: int = 0,
        extra_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        if not linked_memory_indices:
            logger.warning("add_source_cue called with no linked memories. Skipping.")
            return ""

        linked_memory_str = " || ".join(linked_memory_indices)

        existing = self._store.get(source_description)
        if existing and existing.is_cue_index():
            linked_memory_str = combine_list(linked_memory_str, existing.linked_memory)
            logger.info(f"Source cue already exists, merging linked memories: {source_description[:60]}...")

        metadata = {
            "linked_memory": linked_memory_str,
            "cue_type": "source",
            "timestamp_unix": timestamp_unix,
        }
        if data_type:
            metadata["data_type"] = data_type

        if extra_metadata:
            for key, value in extra_metadata.items():
                if value is not None:
                    metadata[key] = value

        rid = self._store.upsert(
            index=source_description,
            value="",
            metadata=metadata,
        )

        logger.info(
            f"Added source cue: '{source_description[:60]}...' "
            f"linking {len(linked_memory_indices)} memories "
            f"(data_type={data_type}, "
            f"extra_fields={list(extra_metadata.keys()) if extra_metadata else []})"
        )

        for primary_index in linked_memory_indices:
            primary_entry = self._store.get(primary_index)
            if primary_entry is None:
                logger.warning(f"Cannot backlink source cue to missing memory: {primary_index}")
                continue

            existing_cues = primary_entry.get_cue_indices()
            if source_description in existing_cues:
                continue
            existing_cues.append(source_description)
            updated_cue_str = "||".join(existing_cues)

            updated_metadata = primary_entry.get_metadata()
            updated_metadata["cue_indices"] = updated_cue_str
            self._store.upsert(
                index=primary_entry.index,
                value=primary_entry.value,
                metadata=updated_metadata,
            )

        return rid

    def _delete_cue_index(self, entry: MemoryEntry) -> None:
        linked_memories = entry.get_linked_memories()
        for primary_index in linked_memories:
            primary_entry = self._store.get(primary_index)
            assert (
                primary_entry is not None
            ), f"Primary entry {primary_index} not found."

            primary_entry.cue_indices = "||".join(
                [ci for ci in primary_entry.get_cue_indices() if ci != entry.index]
            )
            self._store.upsert(
                index=primary_entry.index,
                value=primary_entry.value,
                metadata=primary_entry.get_metadata(),
            )
        self._store.delete(entry.index)

    def _delete_primary_memory(self, entry: MemoryEntry) -> None:
        cue_indices = entry.get_cue_indices()
        for cue_index in cue_indices:
            cue_entry = self._store.get(cue_index)

            if cue_entry is None:
                raise AssertionError(
                    f"Cue entry '{cue_index}' not found. This may indicate a data consistency issue."
                )

            if cue_entry.is_cue_index():
                pass
            elif cue_entry.is_primary_index():

                logger.info(
                    f"Skipping cue index '{cue_index}' during deletion: "
                    f"converted to primary index"
                )
                continue
            else:
                raise AssertionError(
                    f"Cue entry '{cue_index}' is in an invalid state: "
                    f"not a cue index (linked_memory='{cue_entry.linked_memory}') "
                    f"and not a primary index. This requires investigation."
                )

            linked_memories = cue_entry.get_linked_memories()
            linked_memories = [lm for lm in linked_memories if lm != entry.index]
            if linked_memories:
                metadata = cue_entry.get_metadata()
                metadata["linked_memory"] = "||".join(linked_memories)
                self._store.upsert(
                    index=cue_index,
                    value=cue_entry.value,
                    metadata=metadata,
                )
            else:
                self._store.delete(cue_index)
        self._store.delete(entry.index)

    def delete(self, key: str) -> None:
        entry = self._store.get(key)
        if entry is None:
            return

        if entry.is_cue_index():
            self._delete_cue_index(entry)
        elif entry.is_primary_index():
            self._delete_primary_memory(entry)

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        return self._store.list_memories(limit)

    def count(self) -> int:
        return self._store.count()

    def get_backend_type(self) -> str:
        return self.backend_type

    def clear(self) -> None:
        self._store.clear()
