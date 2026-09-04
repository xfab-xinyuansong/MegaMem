"""
MegaMem facade — a single entry point that routes between memory storage backends.
"""

# Standard library
import enum
import logging
import time
from typing import Any, Dict, List, Optional, Union

# Third-party
from omegaconf import DictConfig
from chromadb.api.types import Where

# Local — core components
from megamem.core.base import MemoryBase
from megamem.core.local_memory_store import LocalMemoryStore
from megamem.core.memory_entry import MemoryEntry
from megamem.core.memory_filter import MemoryFilter
from megamem.core.query_generator import QueryGenerator

# Local — utilities
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
    """
    Facade exposing one API across multiple memory storage backends.

    Hides the choice between local persistent storage and HTTP-based storage,
    so callers can swap backends through configuration without changing code.

    Supported backends:
    - 'local': Local persistent storage using ChromaDB
    - 'http': Remote storage via HTTP API calls
    """

    def __init__(self, cfg: DictConfig, user_id: str):
        """Bootstrap the memory facade with the configured backend and helpers."""
        self.cfg = cfg

        # User identifier owning this memory facade.
        self.user_id = user_id

        # Maximum context window we are willing to spend (tokens).
        self.MAX_CONTEXT_TOKENS = 128000  # default large-context budget

        # Wire up the helpers used during memory operations.
        self.query_generator = QueryGenerator(cfg)
        self.memory_filter = MemoryFilter(cfg)  # LLM-driven memory filtering

        # Multimodal toggle, defaulting to enabled.
        self.multimodal_support = cfg.memory.get("multimodal_support", True)

        self._llm_client = ChatCompletionModel(cfg)  # LLM used for update decisions

        # Local persistent ChromaDB-backed store.
        self._store = LocalMemoryStore(cfg, user_id)

        # Pull similarity thresholds from config.
        # These thresholds gate when memories are eligible for updates vs new additions.
        self.QUERY_SCORE_THRESHOLD = (
            cfg.memory.query_score_threshold
        )  # query result threshold

    def get_user_id(self) -> str:
        """
        Return the user_id this memory facade was initialized with.

        Returns:
            str: User ID
        """
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
        extracted: set = set()  # already-emitted memory values for de-duplication

        factual_memory_condition = {"memory_type": {"$eq": "factual"}}

        if query_mode == QueryMode.CUE_ONLY:
            # Restrict to cue-index entries.
            # Note: cue indices don't carry memory_type="factual", so we only filter on linked_memory.
            where = {"linked_memory": {"$ne": ""}}
        elif query_mode == QueryMode.PRIMARY_ONLY:
            # Restrict to primary memory entries.
            query_condition = {"linked_memory": {"$eq": ""}}
            where = {"$and": [query_condition, factual_memory_condition]}

        # Run a search per query variant.
        for query in queries:
            # Vector similarity search through the underlying store.
            results: List[MemoryEntry] = self._store.query(query, top_k, where, include)

            # Build (entry, score) tuples and sort by score descending.
            scored_pairs = [(entry, entry.score) for entry in results]
            scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

            # Threshold + de-duplicate as we walk down the ranked list.
            for entry, score in scored_pairs:

                # Stop once we drop below the configured threshold.
                if score < self.QUERY_SCORE_THRESHOLD:
                    break

                if entry.is_cue_index():
                    # Resolve linked primary memories for this cue.
                    for primary_index in entry.get_linked_memories():
                        primary_entry = self._store.get(primary_index)

                        if not primary_entry:
                            logger.warning(f"Primary memory entry cannot found: {primary_index}")
                            continue

                        # Use the primary entry's value (not the cue's).
                        value = primary_entry.get_memory_value()
                        if value in extracted:
                            continue

                        memory_results.append(primary_entry)  # append the primary entry, not the cue
                        extracted.add(value)
                else:
                    # Primary memory entry path.
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
        """
        Run a hybrid search (keyword or BM25) per the configured method.

        Args:
            context: Search query string
            where: Filter conditions for metadata-based filtering

        Returns:
            List of MemoryEntry objects from hybrid search
        """
        # Look up the hybrid method (defaults to 'bm25').
        hybrid_method = self.cfg.memory.get("hybrid_search_method", "bm25")
        hybrid_top_k = self.cfg.memory.get("hybrid_top_k", 10)

        # Build the BM25 index when needed.
        if hybrid_method == "bm25":
            # Pull the user_id from the where clause for index building.
            target_user_id = extract_user_id_from_where(where) or self.user_id

            # Build the index lazily for this user.
            if target_user_id and target_user_id not in self._store._bm25_indices:
                logger.info(f"Building BM25 index for user {target_user_id} before first search")
                self._store.build_bm25_index(user_id=target_user_id)

        # Dispatch to the chosen hybrid method.
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
        """Delegate to the shared ``merge_with_rrf`` helper."""
        return merge_with_rrf(result_lists, weights=weights, k=k)

    def _search_source_cues(
        self,
        step,
        query_text: str,
        top_k: int,
        latency_tracker,
    ) -> List[MemoryEntry]:
        """SS(source_cues): semantic search over source-cue descriptions."""
        from megamem.core.retrieval_planner import (
            build_where_clause, get_string_filters,
            merge_where_clauses, apply_string_filters,
        )

        # Where clause from the step's metadata (data_type, timestamps).
        step_where = build_where_clause(step)

        # Restrict to source-cue entries (excludes topical cues).
        sc_condition = {"cue_type": {"$eq": "source"}}
        effective_where = merge_where_clauses(step_where, sc_condition)

        # Semantic search over source cue descriptions.
        if latency_tracker:
            with latency_tracker.track("search_source_cues"):
                raw_results = self._store.query(
                    query_text, top_k * 3, effective_where
                )
        else:
            raw_results = self._store.query(
                query_text, top_k * 3, effective_where
            )

        # Score threshold.
        results = [
            entry for entry in raw_results
            if entry.score >= self.QUERY_SCORE_THRESHOLD
        ]

        # Apply the step's string post-filters
        # (sender, recipients, author, title).
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
        """SS(primary_memories): semantic search over extracted facts/details."""
        # Primary memories: not cue indices and tagged factual.
        primary_condition = {"$and": [
            {"linked_memory": {"$eq": ""}},
            {"memory_type": {"$eq": "factual"}},
        ]}

        if step.scope == "filtered_results" and filtered_source_cues:
            # Pattern 5: scope to primary memories linked from the matched source cues.
            linked_ids: set = set()
            for cue in filtered_source_cues:
                linked_ids.update(cue.get_linked_memories())

            if not linked_ids:
                logger.warning(
                    f"SS(primary_memories) {step.step_id}: "
                    f"no linked primary IDs from source cues."
                )
                return []

            # Search wider than top_k, then post-filter to the linked set.
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

        # Pattern 6: search every primary memory (scope=all_sources).
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

    # The 6 valid plan signatures, expressed as tuples of (op, target) per step.
    # target is None for FILTER and RESOLVE.
    _VALID_PLAN_SHAPES = {
        # Pattern 1: FILTER → RESOLVE(metadata_summary)
        (("FILTER", None), ("RESOLVE", None)),

        (("SEMANTIC_SEARCH", "source_cues"), ("RESOLVE", None)),
        # Pattern 5: SS(source_cues) → SS(primary_memories)
        (("SEMANTIC_SEARCH", "source_cues"), ("SEMANTIC_SEARCH", "primary_memories")),
        # Pattern 6: SS(primary_memories) alone
        (("SEMANTIC_SEARCH", "primary_memories"),),
    }

    _VALID_OPS = {"FILTER", "SEMANTIC_SEARCH", "RESOLVE"}
    _VALID_TARGETS = {"source_cues", "primary_memories"}
    _VALID_RETURN_MODES = {"metadata_summary", "full_content"}
    _VALID_SCOPES = {"all_sources", "filtered_results"}
    _VALID_DATA_TYPES = {"mail", "doc", "teams"}

    def _validate_plan(self, plan, context: str) -> List[str]:
        """Validate a RetrievalPlan ahead of execution."""
        issues: List[str] = []
        steps = plan.steps

        # --- 1. Step count ---
        if len(steps) == 0:
            issues.append("ERROR: plan has 0 steps")
            return issues
        if len(steps) > 2:
            issues.append(
                f"ERROR: plan has {len(steps)} steps (max 2). "
                f"Ops: {[s.op for s in steps]}"
            )
            return issues

        # --- 2. Per-step field validation ---
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
                    # Flag source-level fields that get ignored.
                    for fld in ("sender", "recipients", "author", "title", "data_type", "participants", "topic", "conversation_type"):
                        if getattr(step, fld, None):
                            issues.append(
                                f"WARN: {sid} SS(primary_memories) has "
                                f"{fld}='{getattr(step, fld)}' which is ignored "
                                f"(primary memories don't carry source metadata)"
                            )

            # --- data_type guardrail: strip unknown data_type + associated metadata ---
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

        # --- 3. Plan shape validation ---
        shape = tuple(
            (s.op, s.target if s.op == "SEMANTIC_SEARCH" else None)
            for s in steps
        )
        if shape not in self._VALID_PLAN_SHAPES:
            issues.append(
                f"ERROR: unrecognized plan shape {shape}. "
                f"Expected one of: {self._VALID_PLAN_SHAPES}"
            )

        # --- 4. RESOLVE as first step (no source cues to consume) ---
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
        """Execute a planner-driven retrieval pipeline."""
        from megamem.core.retrieval_planner import (
            RetrievalPlanner, build_where_clause,
            get_string_filters, apply_string_filters,
            resolve_source_cues, build_source_cue_filter,
        )

        # --- Plan generation ---
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

        # Surface the plan for debugging / test inspection.
        self._last_plan = plan
        self._last_source_cues = None  # cleared; populated below if SS(source_cues) runs

        # --- Plan validation ---
        issues = self._validate_plan(plan, context)
        for issue in issues:
            logger.warning(f"Plan validation [{context[:40]}...]: {issue}")
        # Bail on any ERROR-level finding — fall back to SS(primary_memories).
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

        # --- Step execution ---
        # State carried between steps:
        filtered_source_cues = None        # produced by FILTER or SS(source_cues)
        memory_results: List[MemoryEntry] = []

        for step in plan.steps:

            if step.op == "FILTER":
                where_clause = build_where_clause(step)
                string_filters = get_string_filters(step)

                # Fetch source cues that match metadata + cue_type="source".
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

            # ----------- SEMANTIC_SEARCH ----------------------
            elif step.op == "SEMANTIC_SEARCH":
                query_text = step.query_text or context

                if step.target == "source_cues":

                    filtered_source_cues = self._search_source_cues(
                        step, query_text, top_k, latency_tracker,
                    )
                    # Mirror to debug attribute for test inspection.
                    self._last_source_cues = filtered_source_cues
                    # Intermediate: consumed by RESOLVE or SS(primary_memories) on the next step.
                    memory_results = filtered_source_cues

                elif step.target == "primary_memories":
                    # SS(primary_memories) — pattern 5 (scoped) or 6 (all)
                    # Semantic search over extracted facts/details.
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
        """Planner-driven retrieval pipeline."""
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
        """Carry out an intelligent semantic search to surface relevant memories."""

        # Normalize the context to a single string.
        context = context_to_str(context)

        # Initialize the result list.
        memory_results = []

        # Track total query time.
        query_start_time = time.time()

        if enhance_query:
            # Have the LLM produce multiple query variants for broader recall —
            # different phrasings often surface different valid hits.
            query_gen_start = time.time()
            queries = self.query_generator.generate_queries(context)
            query_gen_time = time.time() - query_gen_start
            logger.info(f"[LATENCY] Query generation took {query_gen_time:.3f}s")
        else:
            # Skip enhancement and use the raw context as a single query.
            queries = [context]

        # Per-source result buckets used by the RRF merge.
        primary_results = []
        cue_results = []
        hybrid_results = []

        # Search per query mode.
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
            # Pull primary and cue results separately; we merge later.
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

        # If hybrid search is on, run it and RRF-merge every available source.
        if enable_hybrid_search:
            try:
                # Keyword or BM25 search.
                if latency_tracker:
                    with latency_tracker.track("search_hybrid"):
                        hybrid_results = self._perform_hybrid_search(context, where)
                else:
                    hybrid_results = self._perform_hybrid_search(context, where)

                # Stage result lists + weights for RRF.
                result_lists = []
                weights = []

                # Primary semantic search.
                if primary_results:
                    result_lists.append(primary_results)
                    weights.append(2.0)  # heavier weight for primary semantic search

                # Cue-index search.
                if cue_results:
                    result_lists.append(cue_results)
                    weights.append(1.0)  # medium weight for cue index search

                # Hybrid (keyword/BM25) search.
                if hybrid_results:
                    result_lists.append(hybrid_results)
                    weights.append(1.0)  # medium weight for hybrid search

                # Fuse the lists with RRF.
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
                # If hybrid search fails, log and fall back to semantic-only results.
                logger.warning(f"Hybrid search failed: {e}. Falling back to semantic search only.")
                # Fallback: still merge primary and cue results when both exist.
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
            # No hybrid search — only merge primary and cue when in BOTH mode.
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

        # If LLM filtering was explicitly turned on, run it now.
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
        """Augment results with extra memories from each hit's session.

        For every unique (source_conv_idx, source_session) seen in the initial
        results, fetch other primary memories from the same session through
        metadata-only filtering (no embedding needed).

        New memories are appended after the originals (preserving rank order)
        and deduplicated by value.
        """
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
        """Retrieve episodic memories linked from the supplied factual memories."""
        # Collect every unique episodic ID.
        episodic_ids: set = set()
        for entry in memory_results:
            if entry.episodic_memory_ids:
                episodic_ids.update(entry.episodic_memory_ids)

        # Fetch the corresponding episodic memories.
        episodic_memories: Dict[str, MemoryEntry] = {}
        for episodic_id in episodic_ids:
            episodic_entry = self._store.get(episodic_id)
            if episodic_entry:
                episodic_memories[episodic_id] = episodic_entry
            else:
                logger.warning(f"Episodic memory not found: {episodic_id}")

        return episodic_memories

    def get_all_cues(self) -> List[MemoryEntry]:
        """Return every cue-index entry (topical + predictive) in the store."""
        return self._store.get_all_cues()

    def get(self, key: str) -> MemoryEntry:
        """
        Fetch a single record using its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        return self._store.get(key)

    def add(self, entry: MemoryEntry):
        """
        Persist a single memory entry to the store.

        Args:
            entry: MemoryEntry object to add

        Returns:
            Record ID of the added memory entry
        """
        # Cue indices cannot be added manually.
        assert (
            entry.is_primary_index()
        ), "Only primary memory entries can be added directly."

        exist_entry = self._store.get(entry.index)

        # Resolve duplicate indices according to memory type.
        if exist_entry is not None:
            if entry.memory_type == "episodic":
                # Episodic memories are sequential — append (n) so each one is unique.
                # Different episodes may share a similar summary.
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
                # Factual memories: a duplicate is an error.
                raise AssertionError(f"Memory entry {entry.index} already exists.")

        log_memory_operation("Add", entry, user_id=self.user_id)

        # Persist the primary memory.
        self._store.upsert(
            index=entry.index, value=entry.value, metadata=entry.get_metadata()
        )

        # Persist each cue-index entry.
        for cue_index in entry.get_cue_indices():

            cue_entry = self._store.get(cue_index)
            if (cue_entry and cue_entry.is_primary_index()) or cue_index == entry.index:
                # Drop this cue index since it's already a primary index.
                entry.delete_cue_index(cue_index)
                continue

            # Determine the linked memory string.
            linked_memory = entry.index
            if cue_entry and cue_entry.is_cue_index():
                # Combine with whatever this cue already links to.
                linked_memory = combine_list(linked_memory, cue_entry.linked_memory)

            # Insert/update the cue index entry.
            self._store.upsert(
                index=cue_index,
                value="",
                metadata={
                    "linked_memory": linked_memory,
                    "cue_type": "topical",
                },
            )

        # Persist each predictive (extrinsic) cue-index entry.
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
        """Add a source-cue index entry that links to all memories from a source."""
        if not linked_memory_indices:
            logger.warning("add_source_cue called with no linked memories. Skipping.")
            return ""

        # Compose the || -separated linked_memory string.
        linked_memory_str = " || ".join(linked_memory_indices)

        # Source descriptions are stable deduplication keys for the store contract.
        existing = self._store.get(source_description)
        if existing and existing.is_cue_index():
            # Combine with the previous linked-memory list.
            linked_memory_str = combine_list(linked_memory_str, existing.linked_memory)
            logger.info(f"Source cue already exists, merging linked memories: {source_description[:60]}...")

        # Build the source-cue metadata payload.
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

        # Persist the source-cue entry.
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

        # Backlink each primary memory while retaining its complete metadata.
        for primary_index in linked_memory_indices:
            primary_entry = self._store.get(primary_index)
            if primary_entry is None:
                logger.warning(f"Cannot backlink source cue to missing memory: {primary_index}")
                continue

            # Only backlink when the source cue isn't in the primary's cue_indices yet.
            existing_cues = primary_entry.get_cue_indices()
            if source_description in existing_cues:
                continue
            existing_cues.append(source_description)
            updated_cue_str = "||".join(existing_cues)

            # Update the primary memory's metadata in the store.
            updated_metadata = primary_entry.get_metadata()
            updated_metadata["cue_indices"] = updated_cue_str
            self._store.upsert(
                index=primary_entry.index,
                value=primary_entry.value,
                metadata=updated_metadata,
            )

        return rid

    def _delete_cue_index(self, entry: MemoryEntry) -> None:
        """
        Delete a cue index, repairing any primary memories that reference it.

        Args:
            key: Cue index key to delete
        """
        # Walk every primary memory the cue points to.
        linked_memories = entry.get_linked_memories()
        for primary_index in linked_memories:
            primary_entry = self._store.get(primary_index)
            assert (
                primary_entry is not None
            ), f"Primary entry {primary_index} not found."

            # Strip the cue index from the primary memory's cue_indices.
            primary_entry.cue_indices = "||".join(
                [ci for ci in primary_entry.get_cue_indices() if ci != entry.index]
            )
            self._store.upsert(
                index=primary_entry.index,
                value=primary_entry.value,
                metadata=primary_entry.get_metadata(),
            )
        # Finally drop the cue index itself.
        self._store.delete(entry.index)

    def _delete_primary_memory(self, entry: MemoryEntry) -> None:
        """
        Delete a primary memory plus every cue index pointing to it.

        Args:
            entry: Primary memory record to delete
        """
        # Walk each cue index this primary memory references.
        cue_indices = entry.get_cue_indices()
        for cue_index in cue_indices:
            cue_entry = self._store.get(cue_index)

            # Edge case: cue entry doesn't exist.
            if cue_entry is None:
                raise AssertionError(
                    f"Cue entry '{cue_index}' not found. This may indicate a data consistency issue."
                )

            # Branch on the cue entry's actual classification.
            if cue_entry.is_cue_index():
                # Normal path — proceed to delete the cue index.
                pass
            elif cue_entry.is_primary_index():

                logger.info(
                    f"Skipping cue index '{cue_index}' during deletion: "
                    f"converted to primary index"
                )
                continue
            else:
                # Unexpected case: entry is neither cue nor primary — corruption.
                raise AssertionError(
                    f"Cue entry '{cue_index}' is in an invalid state: "
                    f"not a cue index (linked_memory='{cue_entry.linked_memory}') "
                    f"and not a primary index. This requires investigation."
                )

            # Strip this primary index from the cue's linked-memory list.
            linked_memories = cue_entry.get_linked_memories()
            linked_memories = [lm for lm in linked_memories if lm != entry.index]
            if linked_memories:
                # Persist the trimmed links without discarding cue classification
                # or source metadata.
                metadata = cue_entry.get_metadata()
                metadata["linked_memory"] = "||".join(linked_memories)
                self._store.upsert(
                    index=cue_index,
                    value=cue_entry.value,
                    metadata=metadata,
                )
            else:
                # No links remain — drop the cue index entirely.
                self._store.delete(cue_index)
        # Finally drop the primary memory entry itself.
        self._store.delete(entry.index)

    def delete(self, key: str) -> None:
        """
        Remove a record using its natural-language key.

        Args:
            key: Natural language key to delete
        """
        entry = self._store.get(key)
        if entry is None:
            return

        if entry.is_cue_index():
            # Delete the cue index.
            self._delete_cue_index(entry)
        elif entry.is_primary_index():
            # Delete the primary memory and every cue that points to it.
            self._delete_primary_memory(entry)

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        Return up to *limit* memory entries from the collection.

        Args:
            limit: Max number of records to return

        Returns:
            Dict containing memory records
        """
        return self._store.list_memories(limit)

    def count(self) -> int:
        """
        Return the number of records in the collection.

        Returns:
            Number of records in the collection
        """
        return self._store.count()

    def get_backend_type(self) -> str:
        """
        Return the active backend type.

        Returns:
            Backend type ('local' or 'http')
        """
        return self.backend_type

    def clear(self) -> None:
        """
        Drop every record in the collection.
        """
        self._store.clear()
