"""
Dual-query reformulation retrieval.

Asks an LLM to rewrite the user's natural-language question into a
keyword-style search query, then runs TWO searches — one with the rewritten
form, one with the original — and merges the results.

The dual pass exists because a single reformulation tends to drop evidence
that the original phrasing surfaces. Empirically, top-5 Jaccard overlap
between the two query forms is below 0.01, and roughly 38% of failures from
single-pass plan-based retrieval came from the reformulation discarding
the very phrasing that would have hit.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.memory import AgentMemory, QueryMode
from megamem.core.memory_entry import MemoryEntry
from megamem.retriever.base_retriever import BaseMemoryRetriever
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.memory import dedup_memories

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class ReformulatedQuery(BaseModel):
    """LLM-generated optimized search query."""
    search_query: str = Field(
        description=(
            "An optimized search query rewritten from the user's question. "
            "Suitable for semantic similarity search + BM25 keyword matching "
            "against a personal memory database."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was reformulated."
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REFORMULATION_PROMPT = """\
You are a search query optimizer for a personal memory database.

Given a user's natural-language question, rewrite it into an optimized search \
query that will retrieve the most relevant personal memories using semantic \
similarity search combined with BM25 keyword matching.

RULES:
1. Preserve ALL named entities (people, places, dates, events) EXACTLY as written.
2. Convert question syntax to declarative keyword-style for better BM25 matching.
3. Keep the core event or activity description intact — do NOT over-abbreviate.
4. For temporal questions ("when did X happen?"), include the event AND any \
   date/time references mentioned in the question.
5. For counting questions ("how many times…"), include the subject and the \
   countable items.
6. Keep the query between 5 and 20 words. Do NOT drop important nouns or verbs \
   from the original question.
7. Do NOT add information that is not present in the original question.

EXAMPLES:

Question: "What does Rachel do with her family on weekends?"
Search query: "Rachel family weekend activities"

Question: "When did Tom give a presentation at the conference?"
Search query: "Tom gave presentation at conference"

Question: "How many times has Lisa gone skiing in 2023?"
Search query: "Lisa went skiing 2023"

Question: "What book did Emma read from David's recommendation?"
Search query: "Emma read book David recommended"

Question: "Where did Kevin get his cat Whiskers from?"
Search query: "Kevin got cat Whiskers from"

Question: "When did Anna buy a new tank for her fish?"
Search query: "Anna bought new tank fish"

Question: "What is the name of the restaurant Ben tried in March 2023?"
Search query: "Ben restaurant tried March 2023"

Question: "What food was served at the barbecue hosted by Mike on 4 July 2023?"
Search query: "food served barbecue Mike 4 July 2023"

USER QUESTION:
{query}

Produce the optimized search query.\
"""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class QueryReformulationRetriever(BaseMemoryRetriever):
    """
    Rewrite-then-search retriever using a dual-query merge.

    Workflow: an LLM produces a keyword-flavoured rewrite of the user's
    question, both the rewrite and the raw question are sent to the memory
    store, and the two result lists are merged with deduplication.

    Why bother running both queries?
    --------------------------------
    Failure analysis on plan-based retrieval revealed that a rewritten
    question and the original question hit nearly disjoint memory sets —
    Jaccard overlap below 0.01 in the top 5. In about 38% of failures, the
    rewrite stripped the very phrasing that would have surfaced the
    relevant memory. Running both keeps the BM25-friendliness of the
    rewrite while preserving the natural-language semantics of the
    original.

    Example:
        retriever = QueryReformulationRetriever(cfg, memory_client=megamem)
        memories = retriever.retrieve("What is John's favorite food?", top_k=30)
    """

    def __init__(
        self,
        cfg: DictConfig,
        memory_client: Optional[AgentMemory] = None,
        model_client: Optional[ChatCompletionModel] = None,
    ):
        super().__init__(cfg)
        self.memory_client = memory_client
        self.model_client = model_client or ChatCompletionModel(cfg)

        self.top_k = self.cfg.memory.get("top_k", 30)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = self.cfg.retrieval.get("enable_llm_filter", False)

        if self.cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

        self.last_trace: List[Dict] = []

    # ------------------------------------------------------------------
    # Query reformulation
    # ------------------------------------------------------------------

    def _reformulate_query(
        self,
        query: str,
        latency_tracker=None,
    ) -> tuple:
        """
        Run the LLM rewrite step and return ``(search_query, reasoning)``.

        On any failure the method gracefully degrades by returning the
        original query along with a fallback explanation, so callers can
        treat the result as always-valid without extra error handling.
        """
        try:
            llm_start = time.time()
            llm_output: ReformulatedQuery = self.model_client.invoke(
                input=REFORMULATION_PROMPT,
                prompt_args={"query": query},
                response_format=ReformulatedQuery,
                source="QueryReformulationRetriever.reformulate",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("reformulation_llm", llm_duration)

            rewritten = (llm_output.search_query or "").strip() or query
            why = (llm_output.reasoning or "").strip()

            logger.info(
                f"Reformulated: \"{query[:80]}\" → \"{rewritten[:80]}\" "
                f"({why[:60]})"
            )
            return rewritten, why

        except Exception as exc:
            logger.warning(
                f"Query reformulation failed: {exc}. Using original query."
            )
            return query, f"fallback: reformulation failed ({exc})"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_retrieval(
        self,
        query: str,
        reformulated: str,
        reasoning: str,
        reform_count: int,
        orig_count: int,
        final_count: int,
    ) -> None:
        bar = "=" * 70
        block = [
            "",
            bar,
            f"QUERY REFORMULATION RETRIEVER | Original: {query}",
            f"  Reformulated: {reformulated}",
            f"  Reasoning: {reasoning}",
            f"  Reform search: {reform_count} | Orig search: {orig_count} | After dedup: {final_count}",
            bar,
        ]
        logger.info("\n".join(block))

    def _log_final_memories(self, query: str, memories: List[MemoryEntry]) -> None:
        bar = "=" * 70
        block = [
            "",
            bar,
            f"FINAL RESULT | Query: \"{query}\" | {len(memories)} memories returned",
            bar,
        ]
        for pos, mem in enumerate(memories, 1):
            score_str = f" (score={mem.score:.3f})" if mem.score is not None else ""
            block.append(f"  [{pos}]{score_str} {mem.index}: {mem.value or ''}")
        if not memories:
            block.append("  (none)")
        block.append(bar)
        logger.info("\n".join(block))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        enhance_query: bool = False,
        enable_hybrid_search: Optional[bool] = None,
        enable_llm_filter: Optional[bool] = None,
        query_mode: Optional[QueryMode] = None,
        latency_tracker=None,
        **kwargs,
    ) -> List[MemoryEntry]:
        """
        Run dual-query reformulation:

        1. Ask the LLM for an optimized rewrite of the question.
        2. Search the store with the rewritten query.
        3. Search the store with the original query.
        4. Dedup-merge — rewrite results take priority in ordering.
        """
        self.last_trace = []

        if top_k is None:
            top_k = self.top_k
        if enable_hybrid_search is None:
            enable_hybrid_search = self.enable_hybrid_search
        if enable_llm_filter is None:
            enable_llm_filter = self.enable_llm_filter
        if query_mode is None:
            query_mode = self.query_mode

        search_kwargs = dict(
            top_k=top_k,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            query_mode=query_mode,
            latency_tracker=latency_tracker,
        )

        # 1. Rewrite the query.
        reformulated, reasoning = self._reformulate_query(query, latency_tracker)

        # 2. Search with the rewritten query.
        try:
            reform_memories = self.memory_client.query(reformulated, **search_kwargs)
        except Exception as exc:
            logger.error(f"Reformulated query search failed: {exc}")
            reform_memories = []

        # 3. Search with the user's original phrasing.
        try:
            orig_memories = self.memory_client.query(query, **search_kwargs)
        except Exception as exc:
            logger.error(f"Original query search failed: {exc}")
            orig_memories = []

        # 4. Merge — rewrite results come first to bias ordering.
        memories = dedup_memories(reform_memories + orig_memories)

        self._log_retrieval(
            query, reformulated, reasoning,
            len(reform_memories), len(orig_memories), len(memories),
        )
        self._log_final_memories(query, memories)

        self.last_trace.append({
            "action": "REFORMULATE_AND_SEARCH",
            "original_query": query,
            "reformulated_query": reformulated,
            "reasoning": reasoning,
            "reform_memories": len(reform_memories),
            "orig_memories": len(orig_memories),
            "final_memories": len(memories),
        })

        return memories

    def get_trace(self) -> List[Dict]:
        """Return the trace recorded by the most recent ``retrieve()`` call."""
        return self.last_trace
