"""LLM-prompted retrieval policy used as a strong, non-trainable baseline."""

from typing import List, Optional
from omegaconf import DictConfig

from megamem.methods.configs.model_resolver import resolve
from megamem.utils.llm import get_general_chat_completion_client
from megamem.core.memory_entry import MemoryEntry
from .trajectory_utils import RetrievalState, RetrievalAction, ActionType
import logging
import re

logger = logging.getLogger(__name__)

ACTION_SELECTION_PROMPT = """You are a memory retrieval agent. Your task is to select the best action to retrieve relevant memories for answering a question.

## Current State
- Question: {query}
- Memories retrieved so far: {num_retrieved}
- Remaining budget: {budget}

## Already Retrieved Memories
{retrieved_memories}

## Available Actions
{available_actions}

## Instructions
1. Analyze which memory would be most helpful for answering the question
2. Consider diversity - avoid selecting memories too similar to what's already retrieved
3. If you have enough information to answer the question, select STOP
4. Select the action that maximizes information gain

## Output Format
Return ONLY the action number (e.g., "1") or "STOP". No explanation.

Your selection:"""


class LLMPolicy:
    """
    Prompt-driven baseline policy.

        Uses the configured general chat API model to pick
        the next retrieval action; not trainable.
    """

    def __init__(
            self,
            cfg: DictConfig,
            max_primary_actions: int = 5,
            max_cue_actions: int = 3,
        ):
        self.cfg = cfg
        self.client = get_general_chat_completion_client(cfg)
        self.model_name = resolve("chat_low")["model"]
        self.max_primary = max_primary_actions
        self.max_cue = max_cue_actions

        # Last prompt/response retained for debugging only.
        self.last_prompt = None
        self.last_response = None

    def select_action(
        self,
        state: RetrievalState,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
        retrieved_memories: Optional[List[dict]] = None,
    ) -> RetrievalAction:
        """
        Pick the next action by prompting the LLM.

        Args:
            state: Current retrieval state
            primary_candidates: Available primary memory candidates
            cue_candidates: Available cue index candidates
            retrieved_memories: Already retrieved memory contents (for context)

        Returns:
            The chosen :class:`RetrievalAction`.
        """
        # Drop anything already collected.
        already = state.retrieved_memories
        available_primary = [m for m in primary_candidates if m.index not in already]
        available_cue = [m for m in cue_candidates if m.index not in already]

        if state.budget <= 0 or (not available_primary and not available_cue):
            return RetrievalAction(action_type=ActionType.STOP)

        prompt = ACTION_SELECTION_PROMPT.format(
            query=state.query,
            num_retrieved=len(state.retrieved_memories),
            budget=state.budget,
            retrieved_memories=self._format_retrieved(retrieved_memories),
            available_actions=self._format_actions(available_primary, available_cue),
        )

        self.last_prompt = prompt  # keep for debugging

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            selection = response.choices[0].message.content.strip()
            self.last_response = selection

            return self._parse_selection(selection, available_primary, available_cue)

        except Exception as exc:
            logger.warning("LLM policy failed; using greedy retrieval: %s", exc)
            return self._fallback_greedy(available_primary, available_cue)

    def _format_retrieved(self, retrieved_memories: Optional[List[dict]]) -> str:
        """Render up to 5 already-retrieved memories for the prompt header."""
        if not retrieved_memories:
            return "(None yet)"

        # Cap entries to keep the prompt small.
        head = retrieved_memories[:5]
        lines = [f"  {pos}. {(m.get('value', '') or '')[:80]}..." for pos, m in enumerate(head, 1)]

        leftover = len(retrieved_memories) - 5
        if leftover > 0:
            lines.append(f"  ... and {leftover} more")

        return "\n".join(lines)

    def _format_actions(
        self,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> str:
        """Render the candidate actions as a numbered list for the LLM."""

        lines: list = []
        action_idx = 1

        for m in primary_candidates[:self.max_primary]:
            content = (m.value or "")[:100]
            ellipsis = "..." if len(m.value or "") > 100 else ""
            lines.append(
                f"{action_idx}. [PRIMARY] {m.index}\n"
                f"   Content: {content}{ellipsis}"
            )
            action_idx += 1

        for m in cue_candidates[:self.max_cue]:
            lines.append(
                f"{action_idx}. [CUE] {m.index}\n"
                f"   Links to memories about this topic"
            )
            action_idx += 1

        lines.append("STOP. Stop retrieval (enough information gathered)")

        return "\n".join(lines)

    def _parse_selection(
        self,
        selection: str,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> RetrievalAction:
        """Map the LLM's response back into a :class:`RetrievalAction`."""
        selection = selection.strip().upper()

        if "STOP" in selection:
            return RetrievalAction(action_type=ActionType.STOP)

        try:
            # Tolerate minor formatting noise like "1.", "Action 1", etc.
            match = re.search(r'(\d+)', selection)

            if not match:
                raise ValueError("No action number found in {selection}")

            pos = int(match.group()) - 1  # convert to 0-indexed

            num_primary = min(len(primary_candidates), self.max_primary)
            num_cue = min(len(cue_candidates), self.max_cue)

            if 0 <= pos < num_primary:
                m = primary_candidates[pos]
                return RetrievalAction(
                    action_type=ActionType.QUERY_PRIMARY_INDEX,
                    target_memory_index=m.index,
                    score=m.score,
                )

            if pos < num_primary + num_cue:
                m = cue_candidates[pos - num_primary]
                return RetrievalAction(
                    action_type=ActionType.QUERY_CUE_INDEX,
                    target_memory_index=m.index,
                    score=m.score,
                )

            print("=="*10)
            print(f"Falling back to greedy approach due to invalid index {pos + 1}")
            print("=="*10)
            logger.warning(f"Invalid action index {pos + 1}, falling back to greedy")
            return self._fallback_greedy(primary_candidates, cue_candidates)

        except (ValueError, IndexError):
            return self._fallback_greedy(primary_candidates, cue_candidates)

    def _fallback_greedy(
        self,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> RetrievalAction:
        """Greedy fallback: pick the highest-scoring candidate."""
        pool: list = []
        pool.extend((m, ActionType.QUERY_PRIMARY_INDEX) for m in primary_candidates[:self.max_primary])
        pool.extend((m, ActionType.QUERY_CUE_INDEX) for m in cue_candidates[:self.max_cue])

        if not pool:
            return RetrievalAction(action_type=ActionType.STOP)

        best_entry, best_action = max(pool, key=lambda pair: pair[0].score)
        return RetrievalAction(
            action_type=best_action,
            target_memory_index=best_entry.index,
            score=best_entry.score,
        )
