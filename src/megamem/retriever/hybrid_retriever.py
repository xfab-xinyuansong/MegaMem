"""
Hybrid retrieval: reformulation + on-demand decomposition.

A single LLM call both rewrites the user's question into an optimised
search query and decides whether multi-step decomposition is required.
The retriever always performs a dual-pass search (rewritten + original);
when the LLM flags the question as containing an indirect reference that
must first be resolved, it additionally executes a plan-based
decomposition and merges those results.

The point is to let the LLM — not a heuristic — judge complexity, and to
combine the strengths of both strategies:
- Reformulation + dual-pass for evidence recall on simple/temporal queries.
- Plan decomposition for genuinely multi-hop queries with indirect
  entity references.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.memory import AgentMemory, QueryMode
from megamem.core.memory_entry import MemoryEntry
from megamem.retriever.base_retriever import BaseMemoryRetriever
from megamem.retriever.plan_based_retriever import (
    PlanBasedRetriever,
    PlanStep,
    QueryPlan,
)
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.memory import dedup_memories, merge_with_rrf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class HybridQueryResult(BaseModel):
    """Combined reformulation + optional decomposition output."""
    search_query: str = Field(
        description=(
            "An optimized search query rewritten from the user's question. "
            "Always produced regardless of decomposition decision."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of the reformulation and decomposition decision."
    )
    needs_decomposition: bool = Field(
        default=False,
        description=(
            "True ONLY when the query contains an indirect or unknown reference "
            "that must be resolved before the main question can be answered. "
            "False for the vast majority of queries."
        ),
    )
    steps: List[PlanStep] = Field(
        default_factory=list,
        description=(
            "Decomposition plan steps. Empty when needs_decomposition is false. "
            "When present, must contain at least 2 steps with depends_on chains."
        ),
    )
    expanded_queries: List[str] = Field(
        default_factory=list,
        description=(
            "3-5 additional search queries generated through commonsense "
            "reasoning. Each targets a different indirect angle (underlying "
            "needs, emotional context, related domains, implicit references). "
            "Empty when query expansion is not requested."
        ),
    )


class CueScanResult(BaseModel):
    """LLM output selecting relevant cues from the full cue inventory."""
    selected_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of cues that are cognitively relevant to the trigger.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the selected cues are relevant.",
    )


class PostFilterScore(BaseModel):
    """Score for a single memory in post-filter."""
    index: str = Field(description="The memory index (exact text from the list).")
    score: int = Field(
        description="Cognitive relevance: 3=strong link, 2=plausible link, 1=no link.",
        ge=1, le=3,
    )


class PostFilterResponse(BaseModel):
    """LLM output scoring all candidate memories."""
    scores: List[PostFilterScore] = Field(
        description="One score per memory, same order as input."
    )



# ---------------------------------------------------------------------------
# Cue-scan prompt — fed the complete cue inventory for a user
# ---------------------------------------------------------------------------

CUE_SCAN_PROMPT = """\
You are a cognitive memory retrieval specialist. You receive:
1. A user's statement or question (the TRIGGER).
2. A numbered list of memory cues — short phrases summarising facts, habits, \
values, goals, events, or predictive situations from the user's past.

Your task: identify which cues are **cognitively relevant** to the trigger. \
A cue is relevant when it connects to the trigger through ANY of these \
reasoning paths:

- **Contradiction / Value conflict**: the trigger describes behaviour that \
  contradicts or abandons a past value, commitment, or belief referenced by \
  the cue.
- **Causal / Origin**: the cue describes an experience, event, or condition \
  that could have CAUSED or SHAPED the behaviour in the trigger.
- **Forward consequence**: the trigger describes a situation that is a \
  natural downstream consequence of the fact/event in the cue.
- **Emotional echo**: the trigger evokes a feeling that mirrors or revisits \
  an emotional experience captured by the cue.
- **Direct relevance**: the cue and trigger share explicit topical overlap \
  (names, activities, places).

== RULES ==
1. Return the 0-based INDEX numbers of relevant cues in `selected_indices`.
2. Cast a reasonably wide net — include cues that are LIKELY relevant, not \
   just certain matches.  It is much better to include a borderline cue than \
   to miss a truly relevant one.
3. If no cues are relevant, return an empty list.
4. Do NOT fabricate indices that are not in the list.
5. Keep your reasoning brief (1-3 sentences).

== TRIGGER ==
{trigger}

== MEMORY CUES ==
{cue_list}

Select the relevant cue indices.\
"""


# ---------------------------------------------------------------------------
# Post-filter prompt — cognitive relevance scoring
# ---------------------------------------------------------------------------

POST_FILTER_PROMPT = """\
You are a cognitive relevance judge for a personal memory system.

You receive a user's statement (the TRIGGER) and a numbered list of candidate \
memories retrieved from their past conversations. Your job is to score how \
strongly each memory connects to the trigger.

A memory is relevant when it links to the trigger through ANY of:
- **Contradiction / Value conflict**: the trigger opposes or abandons \
  a value, commitment, or habit described in the memory.
- **Causal / Origin**: the memory describes something that could have \
  CAUSED or SHAPED the behaviour in the trigger.
- **Forward consequence**: the trigger is a natural downstream result \
  of the fact/event in the memory.
- **Emotional echo**: the trigger revisits a feeling captured by the memory.
- **Direct topical overlap**: they share a person, activity, or subject.

== SCORING ==
  3 — Strong cognitive link (the memory clearly explains, contradicts, or \
      contextualises the trigger).
  2 — Plausible link (the connection requires a small inferential step \
      but is reasonable).
  1 — No meaningful link (topically unrelated, or the connection is too \
      tenuous to be useful).

== RULES ==
1. Score EVERY memory in the list — do not skip any.
2. Use the EXACT index text from the list for each score entry.
3. Err on the side of keeping memories (prefer 2 over 1 when uncertain).
4. Consider the memory's VALUE text, not just its index title.

== TRIGGER ==
{trigger}

== CANDIDATE MEMORIES ==
{memories_text}

Score all memories.\
"""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

HYBRID_PROMPT = """\
You are a search query optimizer and decomposition planner for a personal \
memory database.

Given a user's input (a question OR a personal statement), you MUST do two things:
1. Rewrite it into an optimized search query (the search_query field).
2. Decide whether multi-step decomposition is needed (the needs_decomposition field).
{expansion_instruction}
== SEARCH QUERY RULES ==
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

== DECOMPOSITION RULES ==
1. Set needs_decomposition to false for MOST queries. A single optimized search \
   handles the vast majority of questions.
2. Set needs_decomposition to true ONLY when the query contains an INDIRECT or \
   UNKNOWN reference that must be resolved before the main question can be \
   answered — e.g. "the team lead's favorite food" requires first resolving \
   who the team lead is.
3. NEVER decompose when the person, event, place, or entity is already named \
   directly in the query.
4. NEVER add a "Who is <name>?" step for people already named in the query.
5. Do NOT decompose when person + event + optional date/time are all explicitly \
   named — a single search covers it.
6. When needs_decomposition is true, produce 2-4 steps with depends_on chains. \
   Use {{<step_id>}} placeholders for values resolved in earlier steps.
7. Order steps so dependencies come first.

== EXAMPLES ==

Question: "What does Rachel do with her family on weekends?"
search_query: "Rachel family weekend activities"
needs_decomposition: false
steps: []

Question: "When did Tom give a presentation at the conference?"
search_query: "Tom gave presentation at conference"
needs_decomposition: false
steps: []

Question: "How many times has Lisa gone skiing in 2023?"
search_query: "Lisa went skiing 2023"
needs_decomposition: false
steps: []

Question: "What food was served at the barbecue hosted by Mike on 4 July 2023?"
search_query: "food served barbecue Mike 4 July 2023"
needs_decomposition: false
steps: []

Question: "What is the project lead's favorite programming language?"
search_query: "project lead favorite programming language"
needs_decomposition: true
steps:
  - S1: query="Who is the project lead?", depends_on=null, purpose="Resolve the project lead's identity"
  - S2: query="{{S1}} favorite programming language", depends_on=S1, purpose="Find their favorite language"

Question: "How old is the CEO's wife?"
search_query: "CEO wife age"
needs_decomposition: true
steps:
  - S1: query="Who is the CEO?", depends_on=null, purpose="Resolve the CEO's identity"
  - S2: query="Who is {{S1}}'s wife?", depends_on=S1, purpose="Resolve the wife's identity"
  - S3: query="How old is {{S2}}?", depends_on=S2, purpose="Find the age"

USER INPUT:
{query}

Produce the search query and decomposition decision.{expansion_suffix}\
"""

_EXPANSION_INSTRUCTION = """
3. Generate 3-5 commonsense-expanded search queries (the expanded_queries field).

"""

_EXPANSION_SUFFIX = " Also produce the expanded queries."

_EXPANSION_RULES = """
== QUERY EXPANSION RULES ==
Generate 3-5 short, focused search queries that target **indirectly relevant** \
memories through commonsense reasoning. Consider:
- **Underlying needs**: practical info that would help respond (e.g., a \
  restaurant question → dietary restrictions, budget, location preferences)
- **Emotional context**: past experiences that might have led to this statement \
  (e.g., feeling lonely → recent social changes, moves, breakups)
- **Related domains**: adjacent topics (e.g., buying a house → financial \
  situation, job stability, family plans)
- **Implicit references**: what the person is likely alluding to without \
  saying directly

Each expansion must bring a genuinely new retrieval direction — do NOT just \
rephrase the original. Keep each expansion between 3 and 12 words.

"""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever(BaseMemoryRetriever):
    """
    Single-LLM-call retriever that fuses query reformulation with
    on-demand plan-based decomposition.

    Flow
    ----
    1. One LLM call yields: optimised ``search_query`` + the
       decomposition decision.
    2. Always: dual-query search (rewritten + original) merged with
       deduplication.
    3. If ``needs_decomposition`` is set: also run the plan steps via a
       composed ``PlanBasedRetriever`` and fold those memories in too.

    Cost model
    ----------
    - Simple queries (~93%): 1 LLM call + 2 searches.
    - Complex queries (~7%): 1 LLM call + 2 searches + N step searches +
      (N-1) extraction LLM calls.

    Example:
        retriever = HybridRetriever(cfg, memory_client=megamem)
        memories = retriever.retrieve("The team lead's favorite food", top_k=30)
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

        self._plan_retriever = PlanBasedRetriever(
            cfg,
            memory_client=memory_client,
            model_client=self.model_client,
        )

        self.top_k = self.cfg.memory.get("top_k", 30)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = self.cfg.retrieval.get("enable_llm_filter", False)
        self.enable_session_expansion = self.cfg.memory.get("enable_session_expansion", False)
        self.session_expansion_k = self.cfg.memory.get("session_expansion_k", 5)
        self.enable_cue_scan = self.cfg.memory.get("enable_cue_scan", False)
        self.enable_post_filter = self.cfg.retrieval.get("enable_post_filter", False)
        self.enable_query_expansion = self.cfg.retrieval.get("enable_query_expansion", False)
        self.query_expansion_top_k = self.cfg.retrieval.get("query_expansion_top_k", None)

        if self.cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

        self._cue_cache: Optional[List[MemoryEntry]] = None

        self.last_trace: List[Dict] = []

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _analyze_query(
        self,
        query: str,
        latency_tracker=None,
    ) -> HybridQueryResult:
        """
        Single LLM round-trip that handles query rewriting, the
        decomposition decision, and (optionally) commonsense-expanded
        retrieval probes — all at once.

        Any failure degrades gracefully: a fallback result echoes the
        original query with no decomposition.
        """
        try:
            if self.enable_query_expansion:
                expansion_instruction = _EXPANSION_INSTRUCTION
                expansion_suffix = _EXPANSION_SUFFIX
            else:
                expansion_instruction = "\n"
                expansion_suffix = ""

            prompt = HYBRID_PROMPT.format(
                query=query,
                expansion_instruction=expansion_instruction,
                expansion_suffix=expansion_suffix,
            )
            if self.enable_query_expansion:
                prompt = prompt + _EXPANSION_RULES

            llm_start = time.time()
            outcome: HybridQueryResult = self.model_client.invoke(
                input=prompt,
                prompt_args={},
                response_format=HybridQueryResult,
                source="HybridRetriever.analyze",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("hybrid_analysis_llm", llm_duration)

            if not (outcome.search_query or "").strip():
                outcome.search_query = query

            if not self.enable_query_expansion:
                outcome.expanded_queries = []

            decomp = "DECOMPOSE" if outcome.needs_decomposition else "SINGLE"
            expansion_info = f", +{len(outcome.expanded_queries)} expansions" if outcome.expanded_queries else ""
            logger.info(
                f"Hybrid [{decomp}]: \"{query[:80]}\" → "
                f"\"{outcome.search_query[:80]}\" "
                f"({outcome.reasoning[:60]}{expansion_info})"
            )
            if outcome.needs_decomposition and outcome.steps:
                for s in outcome.steps:
                    dep = f" (depends_on={s.depends_on})" if s.depends_on else ""
                    logger.info(f"  {s.step_id}: query=\"{s.query}\"{dep}")

            return outcome

        except Exception as exc:
            logger.warning(
                f"Hybrid analysis failed: {exc}. Using original query, no decomposition."
            )
            return HybridQueryResult(
                search_query=query,
                reasoning=f"fallback: analysis failed ({exc})",
                needs_decomposition=False,
                steps=[],
            )

    # ------------------------------------------------------------------
    # Cue-scan: LLM-over-all-cues retrieval
    # ------------------------------------------------------------------

    def _get_cue_inventory(self) -> List[MemoryEntry]:
        """Lazily fetch and cache every cue entry for the current memory store."""
        if self._cue_cache is None:
            try:
                self._cue_cache = self.memory_client.get_all_cues()
            except Exception as exc:
                logger.warning(f"Failed to fetch cue inventory: {exc}")
                self._cue_cache = []
        return self._cue_cache

    def _cue_scan(
        self,
        trigger: str,
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        """Use LLM reasoning to pick relevant cues, then resolve their linked memories.

        Steps:
          1. Build a numbered list of every cue (its index text) plus all
             primary memory indices, so the LLM has full coverage.
          2. Ask the LLM which entries connect to the trigger.
          3. For chosen cue entries, follow their links back to primary
             memories. For chosen primary entries, return them as-is.
        """
        cue_entries = self._get_cue_inventory()
        if not cue_entries:
            logger.info("Cue-scan: no cues in store — skipping")
            return []

        # Numbered catalogue of (display_text, entry_or_None) pairs.
        catalogue: List[tuple] = [(entry.index, entry) for entry in cue_entries]

        # Format for the LLM prompt.
        cue_lines = [f"[{pos}] {text}" for pos, (text, _) in enumerate(catalogue)]
        cue_list_str = "\n".join(cue_lines)

        logger.info(
            f"Cue-scan: {len(catalogue)} cues, "
            f"~{len(cue_list_str) // 4} tokens in prompt"
        )

        try:
            llm_start = time.time()
            scan_result: CueScanResult = self.model_client.invoke(
                input=CUE_SCAN_PROMPT,
                prompt_args={"trigger": trigger, "cue_list": cue_list_str},
                response_format=CueScanResult,
                source="HybridRetriever.cue_scan",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("cue_scan_llm", llm_duration)

            logger.info(
                f"Cue-scan LLM selected {len(scan_result.selected_indices)} cues "
                f"in {llm_duration:.2f}s: {scan_result.reasoning[:120]}"
            )
        except Exception as exc:
            logger.warning(f"Cue-scan LLM call failed: {exc}")
            return []

        # Map chosen indices to actual primary memories.
        resolved: List[MemoryEntry] = []
        seen_indices: set = set()

        for pos in scan_result.selected_indices:
            if pos < 0 or pos >= len(catalogue):
                continue
            _, entry = catalogue[pos]
            if entry is None:
                continue

            if entry.is_cue_index():
                for primary_key in entry.get_linked_memories():
                    if primary_key in seen_indices:
                        continue
                    seen_indices.add(primary_key)
                    try:
                        primary = self.memory_client.get(primary_key)
                        if primary:
                            resolved.append(primary)
                    except Exception:
                        pass
            else:
                if entry.index not in seen_indices:
                    seen_indices.add(entry.index)
                    resolved.append(entry)

        logger.info(
            f"Cue-scan resolved {len(resolved)} unique primary memories "
            f"from {len(scan_result.selected_indices)} selected cues"
        )
        return resolved

    # ------------------------------------------------------------------
    # Post-filter: LLM-based cognitive relevance scoring
    # ------------------------------------------------------------------

    def _post_filter(
        self,
        trigger: str,
        memories: List[MemoryEntry],
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        """Score every candidate memory for cognitive relevance and drop the noise.

        Returns the memories that scored at least 2, sorted by (LLM score
        descending, original position). On any failure the original list
        is returned untouched.
        """
        if not memories:
            return memories

        memories_text = "\n".join(
            f"[{entry.index}]: {entry.get_memory_value()}"
            for entry in memories
        )

        try:
            llm_start = time.time()
            response: PostFilterResponse = self.model_client.invoke(
                input=POST_FILTER_PROMPT,
                prompt_args={
                    "trigger": trigger,
                    "memories_text": memories_text,
                },
                response_format=PostFilterResponse,
                source="HybridRetriever.post_filter",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("post_filter_llm", llm_duration)

            score_map = {item.index: item.score for item in response.scores}

            # Keep memories scoring at least 2 (strong / plausible link).
            # Stamp the cognitive relevance score onto each entry so
            # downstream highlighting can distinguish 3 vs 2.
            kept: List[MemoryEntry] = []
            for entry in memories:
                llm_score = score_map.get(entry.index, 2)  # default keep when missing
                entry.score = float(llm_score)
                if llm_score >= 2:
                    kept.append(entry)

            logger.info(
                f"Post-filter: kept {len(kept)}/{len(memories)} memories "
                f"(scored ≥2) in {llm_duration:.2f}s"
            )
            return kept

        except Exception as exc:
            logger.warning(
                f"Post-filter failed: {exc}. Returning all memories unfiltered."
            )
            return memories

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_retrieval(
        self,
        query: str,
        result: HybridQueryResult,
        reform_count: int,
        orig_count: int,
        plan_count: int,
        final_count: int,
    ) -> None:
        bar = "=" * 70
        decomp = "DECOMPOSE" if result.needs_decomposition else "SINGLE"
        block = [
            "",
            bar,
            f"HYBRID RETRIEVER [{decomp}] | Original: {query}",
            f"  Reformulated: {result.search_query}",
            f"  Reasoning: {result.reasoning}",
            f"  Reform search: {reform_count} | Orig search: {orig_count}"
            + (f" | Plan search: {plan_count}" if plan_count else "")
            + f" | After dedup: {final_count}",
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
        Run the hybrid retrieval pipeline:

        1. Single LLM call: rewrite the query and decide whether to decompose.
        2. Always: dual-query search (rewritten + original).
        3. If decomposition is required: execute plan steps and merge.
        4. Dedup-merge every source list.
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

        # Step 1: single LLM analysis.
        result = self._analyze_query(query, latency_tracker)

        # Step 1.5: search the commonsense-expanded probes from step 1.
        expansion_memories: List[MemoryEntry] = []
        if result.expanded_queries:
            exp_top_k = self.query_expansion_top_k or top_k
            exp_kwargs = dict(search_kwargs, top_k=exp_top_k)
            for eq in result.expanded_queries:
                try:
                    expansion_memories.extend(
                        self.memory_client.query(eq, **exp_kwargs)
                    )
                except Exception as exc:
                    logger.warning(f"Expansion query search failed for '{eq[:60]}': {exc}")

        # Step 2: always-on dual-query search.
        try:
            reform_memories = self.memory_client.query(
                result.search_query, **search_kwargs
            )
        except Exception as exc:
            logger.error(f"Reformulated query search failed: {exc}")
            reform_memories = []

        try:
            orig_memories = self.memory_client.query(query, **search_kwargs)
        except Exception as exc:
            logger.error(f"Original query search failed: {exc}")
            orig_memories = []

        # Step 3: optional plan execution.
        plan_memories: List[MemoryEntry] = []
        if result.needs_decomposition and len(result.steps) > 1:
            plan = QueryPlan(steps=result.steps, reasoning=result.reasoning)

            self._plan_retriever._log_plan(plan, query, 1)

            plan_memories, failed_step, failure_reason = (
                self._plan_retriever._execute_plan(
                    plan,
                    top_k=top_k,
                    enable_hybrid_search=enable_hybrid_search,
                    enable_llm_filter=enable_llm_filter,
                    query_mode=query_mode,
                    latency_tracker=latency_tracker,
                    original_query=query,
                )
            )
            if failed_step:
                logger.warning(
                    f"Plan execution failed at {failed_step}: {failure_reason}. "
                    f"Continuing with dual-query results only."
                )
                plan_memories = []

        # Step 4: cue-scan — let the LLM reason over EVERY cue to find
        # cognitive links the embedding search may have missed.
        cue_scan_memories: List[MemoryEntry] = []
        if self.enable_cue_scan:
            try:
                cue_scan_memories = self._cue_scan(query, latency_tracker)
            except Exception as exc:
                logger.warning(f"Cue-scan failed: {exc}. Continuing without cue-scan.")

        # Step 5: weighted-RRF merge over all source lists.
        # Reformulated > original > cue-scan > expansion = plan.
        source_lists: List[List[MemoryEntry]] = []
        source_weights: List[float] = []
        for lst, w in [
            (reform_memories, 2.0),
            (orig_memories, 1.5),
            (cue_scan_memories, 1.5),
            (expansion_memories, 1.0),
            (plan_memories, 1.0),
        ]:
            if lst:
                source_lists.append(lst)
                source_weights.append(w)

        if source_lists:
            memories = merge_with_rrf(source_lists, weights=source_weights)
        else:
            memories = []

        # Step 5.5: post-filter — LLM scores cognitive relevance, noise drops.
        pre_filter = len(memories)
        if self.enable_post_filter and memories:
            try:
                memories = self._post_filter(query, memories, latency_tracker)
            except Exception as exc:
                logger.warning(f"Post-filter failed: {exc}. Continuing unfiltered.")

        # Step 6: session expansion — pull in additional memories from
        # any session that already had a hit.
        pre_expansion = len(memories)
        if self.enable_session_expansion and memories:
            try:
                memories = self.memory_client.expand_by_session(
                    memories, max_per_session=self.session_expansion_k,
                )
            except Exception as exc:
                logger.warning(f"Session expansion failed: {exc}. Continuing without expansion.")

        # Final sort: session-expanded entries default to score 0, so
        # RRF-scored entries naturally float to the top when truncated.
        memories.sort(key=lambda m: m.score, reverse=True)

        self._log_retrieval(
            query, result,
            len(reform_memories), len(orig_memories),
            len(expansion_memories) + len(plan_memories) + len(cue_scan_memories),
            len(memories),
        )
        self._log_final_memories(query, memories)

        # Record the full trace.
        post_filter_kept = pre_expansion if self.enable_post_filter else pre_filter
        self.last_trace.append({
            "action": "HYBRID_RETRIEVE",
            "original_query": query,
            "reformulated_query": result.search_query,
            "reasoning": result.reasoning,
            "needs_decomposition": result.needs_decomposition,
            "num_plan_steps": len(result.steps),
            "reform_memories": len(reform_memories),
            "orig_memories": len(orig_memories),
            "expansion_memories": len(expansion_memories),
            "plan_memories": len(plan_memories),
            "cue_scan_memories": len(cue_scan_memories),
            "dedup_total": pre_filter,
            "post_filter_kept": post_filter_kept,
            "post_filter_removed": pre_filter - post_filter_kept,
            "session_expansion": len(memories) - pre_expansion,
            "final_memories": len(memories),
        })

        return memories

    def get_trace(self) -> List[Dict]:
        """Return the trace recorded by the most recent ``retrieve()`` call."""
        return self.last_trace
