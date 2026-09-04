"""Plan-based memory retrieval."""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.memory import AgentMemory, QueryMode
from megamem.core.memory_entry import MemoryEntry
from megamem.retriever.base_retriever import BaseMemoryRetriever
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.memory import dedup_memories

logger = logging.getLogger(__name__)

_UNKNOWN = "UNKNOWN"


class PlanStep(BaseModel):
    """A single step in the decomposed retrieval plan."""
    step_id: str = Field(description="Step identifier: 'S1', 'S2', etc.")
    query: str = Field(
        description=(
            "The sub-query to execute against the memory store. "
            "May contain placeholders like {S1} that will be replaced "
            "with the resolved answer from a previous step."
        )
    )
    depends_on: Optional[str] = Field(
        default=None,
        description=(
            "step_id of the step whose answer this query depends on "
            "(e.g. 'S1'). null if the step is independent."
        ),
    )
    purpose: str = Field(description="Brief explanation of what this step resolves.")


class QueryPlan(BaseModel):
    """LLM-generated decomposition plan for a complex query."""
    steps: List[PlanStep] = Field(description="Ordered list of 1-4 steps.")
    reasoning: str = Field(description="Brief explanation of the decomposition strategy.")


class StepAnswer(BaseModel):
    """Concise answer extracted from retrieved memories for one plan step."""
    answer: str = Field(
        description=(
            "A concise factual answer extracted from the retrieved memories. "
            "This will be substituted into subsequent steps' queries. "
            "If the memories do not contain a clear answer, return 'UNKNOWN'."
        )
    )
    confidence: float = Field(default=1.0, description="Confidence 0.0-1.0.")


PLAN_GENERATION_PROMPT = """\
You are a query decomposition planner for a memory retrieval system.

Given a user query, determine whether it can be answered directly with a single \
search, or whether it requires multiple steps where later steps depend on answers \
from earlier ones.

RULES:
1. PREFER a single-step plan. If the query is simple and self-contained, produce a plan with a single step.
2. Only decompose into multiple steps when the query contains an INDIRECT or \
   UNKNOWN reference that must be resolved before the main question can be \
   answered (e.g. "the CEO's wife" needs resolving "the CEO" first because \
   you don't know the CEO's name).
3. NEVER decompose when the query already names the person or entity directly. \
   Do NOT add a "Who is <name>?" step for people already named.
4. Do NOT decompose when the query describes an event, activity, or attribute \
   with specific details already provided (person + event + optional date/time). \
   If the person, event, and any relevant time are all explicitly named, \
   a single search covers it — do NOT add a "locate the event" step first.
5. Each step's query should be a clear, natural search query optimized for \
   semantic retrieval against a memory database. Keep queries short and natural.
6. When a step depends on a previous step's answer, use the placeholder \
   {{<step_id>}} in the query text. The placeholder will be replaced with a \
   SHORT identifier (a name or brief phrase), so write queries expecting that.
7. Keep the number of steps minimal (1-4). Do NOT over-decompose.
8. Order steps so that dependencies are resolved before they are needed.

EXAMPLES:

Query: "What is the capital of France?"
Plan:
- S1: query="capital of France", depends_on=null, purpose="Direct factual lookup"

Query: "What activities does Jack do?"
Plan:
- S1: query="Jack's activities and hobbies", depends_on=null, purpose="Direct lookup of Jack's activities"

Query: "Where did Jack move from?"
Plan:
- S1: query="Where did Jack move from", depends_on=null, purpose="Direct lookup of Jack's origin"

Query: "What food was served at the barbecue hosted by Mike on 4 July 2023?"
Plan:
- S1: query="food served barbecue Mike 4 July 2023", depends_on=null, purpose="Direct lookup — person, event, and date all named"

Query: "How long did the camping trip that Sarah took with her kids last?"
Plan:
- S1: query="Sarah camping trip with kids duration", depends_on=null, purpose="Direct lookup — person and activity already specified"

Query: "What did Alex talk about with his colleague during the video call?"
Plan:
- S1: query="Alex video call colleague discussion topic", depends_on=null, purpose="Direct lookup — person and activity already named"

Query: "The design team lead's favorite restaurant"
Plan:
- S1: query="Who leads the design team?", depends_on=null, purpose="Resolve the design team lead's name"
- S2: query="{{S1}}'s favorite restaurant", depends_on="S1", purpose="Find their favorite restaurant"

Query: "How old is the CEO's wife?"
Plan:
- S1: query="Who is the CEO?", depends_on=null, purpose="Resolve the CEO's identity"
- S2: query="Who is {{S1}}'s wife?", depends_on="S1", purpose="Resolve the wife's identity"
- S3: query="How old is {{S2}}?", depends_on="S2", purpose="Find the age"

USER QUERY:
{query}

Produce the decomposition plan.\
"""

PLAN_RETRY_PROMPT = """\
You are a query decomposition planner for a memory retrieval system.

A previous retrieval plan failed at step {failed_step_id} with reason:
  "{failure_reason}"

Generate a NEW, DIFFERENT plan for the user query below. Avoid the same approach
that led to the failure (e.g. try different phrasings or a simpler decomposition).

{base_rules}

USER QUERY:
{query}

Produce the decomposition plan.\
"""

_BASE_RULES = """\
RULES:
1. PREFER a single-step plan. If the query is simple and self-contained, produce a plan with a single step.
2. Decompose only if an INDIRECT reference must be resolved first (e.g. "the CEO" \
   when you don't know the CEO's name). NEVER add "Who is X?" for named people.
3. Do NOT decompose when person + event + optional date/time are all explicitly named. \
   A single search covers it — do NOT add a "locate the event" step first.
4. Use {{<step_id>}} placeholders for values resolved in earlier steps.
5. Keep steps minimal (1-4). Do NOT over-decompose.
6. Order steps so dependencies come first.\
"""

ANSWER_EXTRACTION_PROMPT = """\
You are an answer extraction assistant. Your answer will be substituted into \
a follow-up search query, so it MUST be extremely concise.

SUB-QUESTION:
{question}

RETRIEVED MEMORIES:
{memories}

INSTRUCTIONS:
- Return ONLY the shortest identifying answer: a proper name, a date, a number, \
  or a very brief noun phrase (1-5 words MAX).
- Good answers: "John Smith", "Sweden", "July 2023", "basketball coach"
- Bad answers: "A married mother with kids who enjoys art" (too long/descriptive)
- If asking "who", return ONLY the person's name (e.g. "Sarah", "Dr. Lee").
- If asking "where", return ONLY the place name.
- If asking "when", return ONLY the date or time reference.
- If the memories do not contain a clear answer, return "UNKNOWN".
- Do NOT add explanation, description, or hedging. Just the shortest answer.\
"""


class PlanBasedRetriever(BaseMemoryRetriever):
    """Retriever that plans a sequence of dependent sub-queries."""

    def __init__(
        self,
        cfg: DictConfig,
        memory_client: Optional[AgentMemory] = None,
        model_client: Optional[ChatCompletionModel] = None,
        max_steps: int = 4,
        max_plan_retries: int = 2,
    ):
        super().__init__(cfg)
        self.memory_client = memory_client
        self.model_client = model_client or ChatCompletionModel(cfg)
        self.max_steps = max_steps
        self.max_plan_retries = max_plan_retries

        self.top_k = self.cfg.memory.get("top_k", 30)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = self.cfg.retrieval.get("enable_llm_filter", False)

        if self.cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

        self.last_trace: List[Dict] = []

    def _log_plan(self, plan: QueryPlan, query: str, attempt: int) -> None:
        """Emit a verbose log of the plan to aid debugging."""
        bar = "=" * 70
        block = [
            "",
            bar,
            f"PLAN-BASED RETRIEVER | Original Query: {query}",
            f"Attempt: {attempt} | Reasoning: {plan.reasoning}",
            bar,
        ]
        for s in plan.steps:
            dep = f" (depends_on={s.depends_on})" if s.depends_on else ""
            block.append(f"  {s.step_id}: query=\"{s.query}\"{dep}")
            block.append(f"         purpose: {s.purpose}")
        block.append(bar)
        logger.info("\n".join(block))

    def _log_step_memories(
        self, step: PlanStep, effective_query: str, memories: List['MemoryEntry'], role: str
    ) -> None:
        """Dump the per-step retrieved memories into the log."""
        bar = "-" * 60
        block = [
            "",
            bar,
            f"STEP {step.step_id} ({role}) | Query: \"{effective_query}\"",
            f"  Purpose: {step.purpose}",
            f"  Retrieved {len(memories)} memories:",
        ]
        for pos, mem in enumerate(memories, 1):
            score_str = f" (score={mem.score:.3f})" if mem.score is not None else ""
            block.append(f"    [{pos}]{score_str} {mem.index}: {mem.value or ''}")
        if not memories:
            block.append("    (none)")
        block.append(bar)
        logger.info("\n".join(block))

    def _log_final_memories(self, query: str, memories: List['MemoryEntry']) -> None:
        """Dump the final returned memories into the log."""
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

    def _generate_plan(
        self,
        query: str,
        failed_step_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        latency_tracker=None,
    ) -> QueryPlan:
        """
        Ask the LLM for a decomposition plan for ``query``.

        On the first attempt the standard prompt is used. On subsequent
        retries a different "retry" prompt is used so the LLM is
        explicitly steered away from the strategy that just failed.
        """
        if failed_step_id and failure_reason:
            prompt = PLAN_RETRY_PROMPT.format(
                failed_step_id=failed_step_id,
                failure_reason=failure_reason,
                base_rules=_BASE_RULES,
                query=query,
            )
            prompt_args = None
        else:
            prompt = PLAN_GENERATION_PROMPT
            prompt_args = {"query": query}

        try:
            llm_start = time.time()
            plan: QueryPlan = self.model_client.invoke(
                input=prompt,
                prompt_args=prompt_args,
                response_format=QueryPlan,
                source="PlanBasedRetriever.plan",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("plan_generation_llm", llm_duration)

            if len(plan.steps) > self.max_steps:
                logger.warning(
                    f"Plan has {len(plan.steps)} steps, truncating to {self.max_steps}"
                )
                plan.steps = plan.steps[: self.max_steps]

            return plan

        except Exception as exc:
            logger.error(f"Plan generation failed: {exc}. Falling back to single-step.")
            return QueryPlan(
                steps=[
                    PlanStep(
                        step_id="S1",
                        query=query,
                        depends_on=None,
                        purpose="Fallback: direct search (plan generation failed)",
                    )
                ],
                reasoning=f"Fallback: plan generation failed ({exc})",
            )

    @staticmethod
    def _collapse_independent_steps(plan: QueryPlan, original_query: str) -> QueryPlan:
        """
        Fold a fully-independent multi-step plan into a single step.

        When every step has ``depends_on=None``, the parallel sub-queries
        cover overlapping semantic space — running them all just adds
        retrieval cost without adding information. A single search using
        the user's original query gives equivalent coverage.

        Plans containing at least one dependency are returned as-is.
        """
        if len(plan.steps) <= 1:
            return plan

        if any(s.depends_on for s in plan.steps):
            return plan

        logger.info(
            f"Collapsing {len(plan.steps)} independent steps into a "
            f"single-step plan for query: '{original_query[:80]}'"
        )
        return QueryPlan(
            steps=[
                PlanStep(
                    step_id="S1",
                    query=original_query,
                    depends_on=None,
                    purpose=(
                        f"Collapsed from {len(plan.steps)} independent "
                        f"sub-queries into a single direct search"
                    ),
                )
            ],
            reasoning=(
                f"Original plan had {len(plan.steps)} steps with no "
                f"inter-step dependencies — collapsed to single search. "
                f"Original reasoning: {plan.reasoning}"
            ),
        )

    def _extract_answer(
        self,
        question: str,
        memories: List[MemoryEntry],
        latency_tracker=None,
    ) -> Tuple[str, float]:
        """
        Pull a short answer out of ``memories`` for substitution into
        downstream step queries.

        Returns
        -------
        (answer, confidence)
            answer:     extracted string or _UNKNOWN
            confidence: 0.0-1.0 confidence score (0.0 for UNKNOWN / errors)
        """
        if not memories:
            return _UNKNOWN, 0.0

        # Trim to the strongest hits before sending to the LLM.
        top_memories = memories[:10]
        memories_text = "\n".join(
            f"[{pos}] {mem.index}: {mem.value or ''}"
            for pos, mem in enumerate(top_memories, 1)
        )

        try:
            llm_start = time.time()
            extracted: StepAnswer = self.model_client.invoke(
                input=ANSWER_EXTRACTION_PROMPT,
                prompt_args={"question": question, "memories": memories_text},
                response_format=StepAnswer,
                source="PlanBasedRetriever.extract",
            )
            llm_duration = time.time() - llm_start

            if latency_tracker:
                latency_tracker.add_timing("answer_extraction_llm", llm_duration)

            answer = (extracted.answer or "").strip() or _UNKNOWN
            confidence = extracted.confidence if answer != _UNKNOWN else 0.0
            logger.info(
                f"Extracted answer for '{question[:60]}': "
                f"'{answer}' (conf={confidence:.2f})"
            )
            return answer, confidence

        except Exception as exc:
            logger.error(f"Answer extraction failed: {exc}")
            return _UNKNOWN, 0.0

    @staticmethod
    def _substitute_placeholders(template: str, resolved: Dict[str, str]) -> str:
        """Swap ``{S1}``, ``{S2}`` … placeholders for the resolved answer strings."""
        rendered = template
        for step_id, answer in resolved.items():
            rendered = rendered.replace("{" + step_id + "}", answer)
        return rendered

    @staticmethod
    def _extract_entity_from_query(original_query: str, step_query: str) -> str:
        """
        Recover the entity reference from the original user query that the
        pointer step was trying to resolve. Used as a degraded fallback
        when the LLM cannot extract an answer.

        Example: original query "What country is Caroline's grandma from?"
        and step query "Who is Caroline's grandma?" → "Caroline's grandma".
        """
        possessive_match = re.search(
            r"(\w+(?:'s|'s)\s+\w+(?:\s+\w+)?)", original_query
        )
        if possessive_match:
            return possessive_match.group(1)
        return step_query

    def _execute_plan(
        self,
        plan: QueryPlan,
        top_k: int,
        enable_hybrid_search: bool,
        enable_llm_filter: bool,
        query_mode: QueryMode,
        latency_tracker,
        original_query: str = "",
    ) -> Tuple[List[MemoryEntry], Optional[str], Optional[str]]:
        """Run every step of ``plan`` in order."""
        # Pre-compute the set of step ids that have at least one dependent.
        step_ids_with_dependents: set = {
            s.depends_on for s in plan.steps if s.depends_on
        }

        resolved_answers: Dict[str, str] = {}
        leaf_memories: List[MemoryEntry] = []

        for step in plan.steps:
            step_start = time.time()
            is_pointer = step.step_id in step_ids_with_dependents

            effective_query = self._substitute_placeholders(
                step.query, resolved_answers
            )

            # ---- Run this step's query against the memory store ----
            step_memories: List[MemoryEntry] = []
            query_error: Optional[str] = None

            try:
                step_memories = self.memory_client.query(
                    effective_query,
                    top_k=top_k,
                    enable_hybrid_search=enable_hybrid_search,
                    enable_llm_filter=enable_llm_filter,
                    query_mode=query_mode,
                    latency_tracker=latency_tracker,
                )
            except Exception as exc:
                query_error = str(exc)
                logger.error(
                    f"Step {step.step_id}: query raised exception: {exc}"
                )

            # ---- Pointer-step failure handling ----
            if is_pointer:
                self._log_step_memories(step, effective_query, step_memories, "pointer")

                if query_error:
                    failure_reason = (
                        f"query exception at pointer step {step.step_id}: {query_error}"
                    )
                    self._record_step(
                        step, effective_query, resolved_answers,
                        step_memories, "FAILED_EXCEPTION",
                        time.time() - step_start, latency_tracker,
                    )
                    return [], step.step_id, failure_reason

                if not step_memories:
                    failure_reason = (
                        f"pointer step {step.step_id} returned no memories "
                        f"for query '{effective_query[:80]}'"
                    )
                    logger.warning(f"Step {step.step_id}: {failure_reason}")
                    self._record_step(
                        step, effective_query, resolved_answers,
                        step_memories, "FAILED_EMPTY",
                        time.time() - step_start, latency_tracker,
                    )
                    return [], step.step_id, failure_reason

                # Resolve a short answer for downstream substitution.
                answer, confidence = self._extract_answer(
                    effective_query, step_memories, latency_tracker
                )
                logger.info(
                    f"  STEP {step.step_id} extracted answer: \"{answer}\" "
                    f"(conf={confidence:.2f})"
                )

                if answer == _UNKNOWN:
                    # Confidence gating: degrade rather than hard-fail.
                    fallback = self._extract_entity_from_query(
                        original_query, effective_query
                    )
                    logger.warning(
                        f"Step {step.step_id}: pointer could not resolve "
                        f"'{effective_query[:80]}' — degrading to fallback "
                        f"substitution '{fallback}' and continuing plan"
                    )
                    resolved_answers[step.step_id] = fallback
                    # Keep the pointer's memories since otherwise the
                    # context they provide is lost.
                    leaf_memories.extend(step_memories)

                    self._record_step(
                        step, effective_query, resolved_answers,
                        step_memories, "DEGRADED_UNKNOWN",
                        time.time() - step_start, latency_tracker,
                    )
                else:
                    resolved_answers[step.step_id] = answer

            else:
                # Leaf step — empty hits aren't fatal, just less coverage.
                if query_error:
                    logger.warning(
                        f"Step {step.step_id} (leaf): query error '{query_error}' — "
                        f"skipping memories for this step"
                    )
                else:
                    leaf_memories.extend(step_memories)

                self._log_step_memories(step, effective_query, step_memories, "leaf")

            self._record_step(
                step, effective_query, resolved_answers,
                step_memories, "OK",
                time.time() - step_start, latency_tracker,
            )

        return dedup_memories(leaf_memories), None, None

    def _record_step(
        self,
        step: PlanStep,
        effective_query: str,
        resolved_answers: Dict[str, str],
        memories: List[MemoryEntry],
        status: str,
        duration: float,
        latency_tracker,
    ) -> None:
        """Append one step's outcome to the running trace."""
        step_data = {
            "step": step.step_id,
            "action": "PLAN_STEP",
            "status": status,
            "original_query": step.query,
            "effective_query": effective_query,
            "purpose": step.purpose,
            "depends_on": step.depends_on,
            "resolved_answer": resolved_answers.get(step.step_id, ""),
            "memories_retrieved": len(memories),
            "duration": duration,
        }
        self.last_trace.append(step_data)
        if latency_tracker:
            latency_tracker.add_retrieval_step(step_data)

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
        """Retrieve memories using plan-based decomposition with automatic"""
        self.last_trace = []

        if top_k is None:
            top_k = self.top_k
        if enable_hybrid_search is None:
            enable_hybrid_search = self.enable_hybrid_search
        if enable_llm_filter is None:
            enable_llm_filter = self.enable_llm_filter
        if query_mode is None:
            query_mode = self.query_mode

        failed_step_id: Optional[str] = None
        failure_reason: Optional[str] = None

        for attempt in range(self.max_plan_retries + 1):
            # Plan (or re-plan with failure context).
            plan = self._generate_plan(
                query,
                failed_step_id=failed_step_id,
                failure_reason=failure_reason,
                latency_tracker=latency_tracker,
            )

            # Fold fully-independent multi-step plans into a single search.
            plan = self._collapse_independent_steps(plan, query)

            self._log_plan(plan, query, attempt + 1)

            plan_trace = {
                "step": 0,
                "action": "PLAN_GENERATED",
                "attempt": attempt + 1,
                "num_steps": len(plan.steps),
                "reasoning": plan.reasoning,
                "retry_due_to": failure_reason or "",
                "steps": [
                    {
                        "step_id": s.step_id,
                        "query": s.query,
                        "depends_on": s.depends_on,
                    }
                    for s in plan.steps
                ],
            }
            self.last_trace.append(plan_trace)
            if latency_tracker:
                latency_tracker.add_retrieval_step(plan_trace)

            memories, failed_step_id, failure_reason = self._execute_plan(
                plan,
                top_k=top_k,
                enable_hybrid_search=enable_hybrid_search,
                enable_llm_filter=enable_llm_filter,
                query_mode=query_mode,
                latency_tracker=latency_tracker,
                original_query=query,
            )

            if failed_step_id is None:

                if len(plan.steps) > 1:
                    try:
                        supplement = self.memory_client.query(
                            query,
                            top_k=top_k,
                            enable_hybrid_search=enable_hybrid_search,
                            enable_llm_filter=enable_llm_filter,
                            query_mode=query_mode,
                            latency_tracker=latency_tracker,
                        )
                        memories = dedup_memories(memories + supplement)
                        logger.info(
                            f"Supplemented multi-step results with "
                            f"{len(supplement)} direct-search memories "
                            f"(total after dedup: {len(memories)})"
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Direct-search supplement failed: {exc}"
                        )

                self._log_final_memories(query, memories)
                return memories

            # This attempt failed; either retry or fall back.
            if attempt < self.max_plan_retries:
                logger.warning(
                    f"Plan attempt {attempt + 1}/{self.max_plan_retries + 1} failed "
                    f"at step {failed_step_id}: {failure_reason}. Re-generating plan."
                )
            else:
                logger.warning(
                    f"All {self.max_plan_retries + 1} plan attempts failed. "
                    f"Falling back to direct single-step search for '{query[:60]}'"
                )
                try:
                    fallback_memories = self.memory_client.query(
                        query,
                        top_k=top_k,
                        enable_hybrid_search=enable_hybrid_search,
                        enable_llm_filter=enable_llm_filter,
                        query_mode=query_mode,
                        latency_tracker=latency_tracker,
                    )
                except Exception as exc:
                    logger.error(f"Fallback direct search also failed: {exc}")
                    fallback_memories = []

                self.last_trace.append({
                    "step": 0,
                    "action": "FALLBACK_DIRECT_SEARCH",
                    "query": query,
                    "memories_retrieved": len(fallback_memories),
                })
                self._log_final_memories(query, fallback_memories)
                return fallback_memories

        # Defensive: the loop above should always return.
        return []

    def get_trace(self) -> List[Dict]:
        """Return the trace recorded by the most recent ``retrieve()`` call."""
        return self.last_trace
