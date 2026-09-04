"""
LLM-prompted iterative retrieval policy.

Builds on a base semantic lookup by letting an LLM act as a control loop —
deciding at each step whether to expand into linked memories, reformulate
and re-query the store, or stop because the working set is sufficient.
"""

import time
import json
import logging
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from megamem.retriever.base_retriever import BaseMemoryRetriever
from megamem.core.memory_expander import MemoryExpander
from megamem.core.memory_entry import MemoryEntry
from megamem.core.memory import AgentMemory, QueryMode
from megamem.utils.memory import dedup_memories
from megamem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


LLM_POLICY_PROMPT = """
SYSTEM:
You are MemoryControl, a decision policy for an iterative memory system. Your goal is to help retrieve relevant memories to answer user questions.

Initial retrieval has ALREADY been performed, yielding a Working Set (W) of memories and a Frontier (F) of expansion candidates.
Your job is to decide how to proceed using ONLY these actions:
- EXPAND: grow memory from expansion candidates in Frontier if they add useful information to answering the user question.
- RE_QUERY: regenerate a better query and retrieve again, if current memories are insufficient to answer the question, or important facts are missing.
- STOP: enough information has been collected in the Working Set (W) to answer the user question.

Definitions:
- Working Set (W): memories already collected
- Frontier (F): expansion candidates reachable from W
- Expansion is cheaper than re-querying the full corpus

Decision rules:
1) Choose STOP if W is sufficient to answer the user question with high confidence.
2) Choose EXPAND if frontier contains high-novelty items likely to fill remaining gaps.
3) Choose RE_QUERY if important facts are missing AND expansion is unlikely to find them.
4) Here are some typical scenarios for RE_QUERY:
    (a) Query refinement: The previous query failed to return relevant results, or the question requires a different semantic angle.
    (b) Relative answers: If W provides a relative answer (a "pointer") rather than a direct value. You need to RE-QUERY for the specific target.
        - *Example:*
            - Query: "Where did Mike go to college?"
            - W: "Mike went to the same college as Sarah"  (relative answer, no specific college - gap identified)
            - Action: RE_QUERY -> `new_query`: "Where did Sarah go to college?" 
        - *Example:*
            - Query: "When is the deadline for Project X?"
            - W: "The deadline is three months after the project kickoff." (relative answer, no specific date - gap identified)
            - Action: RE_QUERY -> `new_query`: "When was the kickoff date for Project X?"
5) Prefer EXPAND over RE_QUERY when both are viable.
6) Minimize redundancy and unnecessary steps.

Output STRICT JSON. No extra text.

JSON schema:
{{
  'action': 'EXPAND | RE_QUERY | STOP',
  'reason': 'one sentence',
  'confidence': 0.0-1.0,
  'frontier_ids': ['id1','id2'],  // required if EXPAND, pick from frontier
  'new_query': 'string',          // required if RE_QUERY
}}

UserQuestion:
{user_question}

CurrentQuery:
{current_query}

Step:
{step}/{max_steps}

WorkingSetSummary:
{W_summary}

FrontierSummary:
{F_summary}

Running History:
{trace}

Constraints:
- Prefer EXPAND over RE_QUERY when possible
- Avoid repeating information already covered in W
- RE_QUERY should be concise and retrieval-optimized
- Avoid RE_QUERY with the same query as before
"""

class PromptedPolicyRetriever(BaseMemoryRetriever):
    """Multi-step retriever driven by an LLM-prompted control policy."""

    def __init__(
        self, 
        cfg: DictConfig,
        memory_client: Optional[AgentMemory] = None,
        model_client: Optional[ChatCompletionModel] = None,
        max_steps: int = 5,
    ):
        """
        Build the policy-driven retriever.

        Args:
            cfg: Configuration object
            memory_client: Optional pre-initialized memory client
            model_client: Optional LLM client for policy decisions
            max_steps: Maximum retrieval iterations (default: 5)
        """
        super().__init__(cfg)
        self.memory_client = memory_client
        self.model_client = model_client or ChatCompletionModel(cfg)

        # Pull retrieval knobs from the shared config.
        self.top_k = self.cfg.memory.get("top_k", 10)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.enable_llm_filter = self.cfg.retrieval.get("enable_llm_filter", False)

        if self.cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

        # Build the expander; relaxed-frontier and worker count both come
        # from config (worker count is shared with eval).
        enable_relaxed_frontier = self.cfg.retrieval.get("enable_relaxed_frontier", False)
        max_workers = self.cfg.eval.get("max_workers", 5)

        self.expander = MemoryExpander(
            memory_client=memory_client,
            enable_relaxed_frontier=enable_relaxed_frontier,
            max_workers=max_workers
        )
        self.max_steps = max_steps

        self.last_trace = []

    def _select_from_frontier(
            self,
            frontier: Dict[str, MemoryEntry],
            frontier_ids: List[str] = None,
    ) -> List[MemoryEntry]:
        """
        Translate a list of frontier IDs requested by the LLM into the
        actual ``MemoryEntry`` objects to add to the working set.

        Args:
            frontier: Current frontier dictionary
            frontier_ids: Specific IDs requested by LLM
            max_expand: Maximum items to expand if no specific IDs given

        Returns:
            List of MemoryEntry objects to add
        """
        if not frontier_ids:
            return []

        picked: List[MemoryEntry] = []
        for fid in frontier_ids:
            if fid in frontier:
                picked.append(frontier[fid])
        return picked

    def expand(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Walk one hop along ``links`` for each memory and append the
        resulting entries to the input list.

        Args:
            memories: List of MemoryEntry objects to expand

        Returns:
            Expanded list of MemoryEntry objects
        """
        expanded_memories = memories.copy()
        for memory in memories:
            if memory.links:
                linked_memories = self.memory_client.get_by_ids(memory.links)
                expanded_memories.extend(linked_memories)
        return expanded_memories

    def _format_working_set(self, memories: List[MemoryEntry]) -> str:
        """Render the working set as a numbered, length-capped list for the prompt."""
        if not memories:
            return "(empty)"

        rows = []
        for pos, mem in enumerate(memories[:]):
            # Cap each value so the prompt stays compact.
            value = (mem.value or "")[:150]
            rows.append(
                f"[{pos+1}] {mem.index}: {value}{'...' if len(mem.value or '') > 150 else ''}"
            )

        return "\n".join(rows)

    def _format_frontier(self, frontier: Dict[str, MemoryEntry]) -> str:
        """Render the frontier candidates as bullet points for the prompt."""
        if not frontier:
            return "(empty - no expansion candidates)"

        rows = []
        for mem_id, mem in frontier.items():
            value = (mem.value or "")[:100]
            rows.append(
                f"- [{mem_id}]: {value}{'...' if len(mem.value or '') > 100 else ''}"
            )

        return "\n".join(rows)

    def prompted_policy(
        self,
        user_question: str,
        current_query: str,
        memory_entries: List[MemoryEntry],
        frontier: Dict[str, MemoryEntry],
        step: int,
        trace: Optional[List] = None,
        latency_tracker = None,
    ) -> Dict[str, Any]:
        """Ask the LLM what action to take next given the current state."""

        W_summary = self._format_working_set(memory_entries)
        F_summary = self._format_frontier(frontier)

        # Stringify the running trace for the prompt.
        trace_str = "\n".join([str(t) for t in trace]) if trace else "No history yet"

        prompt = LLM_POLICY_PROMPT.format(
            user_question=user_question,
            current_query=current_query,
            step=step,
            max_steps=self.max_steps,
            W_summary=W_summary,
            F_summary=F_summary,
            trace=trace_str,
        )

        try:
            llm_start = time.time()
            response = self.model_client.invoke(
                input=prompt,
                temperature=0.0,
                source="PromptedPolicyRetriever"
            )
            llm_duration = time.time() - llm_start

            # Strip optional markdown fencing before JSON parsing.
            if isinstance(response, str):
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()

                decision = json.loads(response)
            else:
                decision = response

            # Default to STOP when the LLM omits the field.
            if "action" not in decision:
                decision["action"] = "STOP"

            # Surface LLM latency back through the decision dict.
            decision["llm_duration"] = llm_duration

            if latency_tracker:
                latency_tracker.add_timing("policy_llm", llm_duration)

            return decision

        except Exception as e:
            logger.error(f"Error in prompted_policy: {e}")
            return {
                "action": "STOP",
                "reason": "Error in policy decision",
                "confidence": 0.0,
                "llm_duration": 0.0
            }

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        enhance_query: bool = False,
        enable_hybrid_search: Optional[bool] = None,
        enable_llm_filter: Optional[bool] = None,
        query_mode: Optional[QueryMode] = None,
        latency_tracker = None,
        **kwargs
    ) -> List[MemoryEntry]:
        """Run the iterative LLM-driven retrieval loop."""
        # Reset per-call state on the expander and trace.
        self.expander.reset()
        self.last_trace = []

        # Fall back to config defaults for any override left as None.
        if top_k is None:
            top_k = self.top_k
        if enable_hybrid_search is None:
            enable_hybrid_search = self.enable_hybrid_search
        if enable_llm_filter is None:
            enable_llm_filter = self.enable_llm_filter
        if query_mode is None:
            query_mode = self.query_mode

        # Step 0: required first-pass retrieval against the store.
        step_start = time.time()

        memory_entries = self.memory_client.query(
            query,
            top_k=top_k,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            query_mode=query_mode,
            latency_tracker=latency_tracker,
        )

        current_query = query
        frontier: Dict[str, MemoryEntry] = {}
        frontier = self.expander.build_frontier(frontier, memory_entries)

        step_duration = time.time() - step_start

        step_data = {
            "step": 0,
            "action": "INIT_RETRIEVE",
            "query": query,
            "memories_count": len(memory_entries),
            "frontier_size": len(frontier),
            "duration": step_duration,
        }

        if latency_tracker:
            search_breakdown = latency_tracker.get_search_breakdown()
            if search_breakdown:
                step_data["search_components"] = search_breakdown

        self.last_trace.append(step_data)

        if latency_tracker:
            latency_tracker.add_retrieval_step(step_data)

        logger.info(f"Initial retrieval: {len(memory_entries)} memories, {len(frontier)} frontier candidates")

        for step_idx in range(1, self.max_steps + 1):
            step_start = time.time()

            decision = self.prompted_policy(
                user_question=query,
                current_query=current_query,
                memory_entries=memory_entries,
                frontier=frontier,
                step=step_idx,
                trace=self.last_trace,
                latency_tracker=latency_tracker
            )

            action = decision.get("action", "STOP")

            # ---- STOP ----
            if action == "STOP":
                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "STOP",
                    "reason": decision.get("reason", ""),
                    "confidence": decision.get("confidence", 0.0),
                    "duration": step_duration,
                    "llm_duration": decision.get("llm_duration", 0.0),
                }
                self.last_trace.append(step_data)

                if latency_tracker:
                    latency_tracker.add_retrieval_step(step_data)

                logger.info(f"Step {step_idx}: STOP - {decision.get('reason', '')}")
                break

            # ---- EXPAND ----
            if action == "EXPAND":
                frontier_ids = decision.get("frontier_ids", [])
                chosen = self._select_from_frontier(frontier, frontier_ids)

                print(f"Step {step_idx}: EXPAND - chosen {len(chosen)} from frontier")
                print("Chosen memories:")
                for mem in chosen:
                    print(f"- [{mem.index}]: {mem.value[:100]}{'...' if len(mem.value or '') > 100 else ''}")

                if chosen:
                    # Fold the chosen items into the working set.
                    memory_entries = dedup_memories(memory_entries + chosen)

                    # Drop them from the frontier so they aren't re-picked.
                    for mem in chosen:
                        frontier.pop(mem.index, None)

                    # Re-grow the frontier from the newly added memories.
                    frontier = self.expander.build_frontier(frontier, chosen)

                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "EXPAND",
                    "added": len(chosen),
                    "frontier_ids": frontier_ids,
                    "new_frontier_size": len(frontier),
                    "duration": step_duration,
                    "llm_duration": decision.get("llm_duration", 0.0),
                }
                self.last_trace.append(step_data)

                if latency_tracker:
                    latency_tracker.add_retrieval_step(step_data)

                logger.info(f"Step {step_idx}: EXPAND - added {len(chosen)} memories")
                continue

            # ---- RE_QUERY ----
            if action == "RE_QUERY":
                new_query = decision.get("new_query", current_query)
                current_query = new_query

                new_entries = self.memory_client.query(
                    current_query,
                    top_k=top_k,
                    enable_hybrid_search=enable_hybrid_search,
                    enable_llm_filter=enable_llm_filter,
                    query_mode=query_mode,
                    latency_tracker=latency_tracker,
                )

                # Merge the fresh hits into the working set.
                memory_entries = dedup_memories(memory_entries + new_entries)

                # And refresh the frontier from those new hits.
                frontier = self.expander.build_frontier(frontier, new_entries)

                step_duration = time.time() - step_start
                step_data = {
                    "step": step_idx,
                    "action": "RE_QUERY",
                    "new_query": new_query,
                    "new_memories": len(new_entries),
                    "total_memories": len(memory_entries),
                    "duration": step_duration,
                    "llm_duration": decision.get("llm_duration", 0.0),
                }

                if latency_tracker:
                    search_breakdown = latency_tracker.get_search_breakdown()
                    if search_breakdown:
                        step_data["search_components"] = search_breakdown

                self.last_trace.append(step_data)

                if latency_tracker:
                    latency_tracker.add_retrieval_step(step_data)

                logger.info(f"Step {step_idx}: RE_QUERY - '{new_query}' got {len(new_entries)} new memories")
                continue

            # ---- Unknown action — record and break out. ----
            step_duration = time.time() - step_start
            step_data = {
                "step": step_idx,
                "action": "INVALID",
                "decision": decision,
                "duration": step_duration,
            }
            self.last_trace.append(step_data)

            if latency_tracker:
                latency_tracker.add_retrieval_step(step_data)

            logger.warning(f"Step {step_idx}: Invalid action '{action}', stopping")
            break

        logger.info(f"Retrieval complete: {len(memory_entries)} memories in {len(self.last_trace)} steps")
        return memory_entries

    def get_trace(self) -> List[Dict]:
        """Return the per-step trace from the most recent retrieval."""
        return self.last_trace
