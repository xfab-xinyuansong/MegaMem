import logging
from typing import List

from omegaconf import DictConfig
from pydantic import BaseModel, Field
from megamem.utils.llm import ChatCompletionModel
from megamem.core.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


# Pydantic schemas describing the structured LLM output
class MemoryScore(BaseModel):
    """One memory's relevance verdict."""
    index: str = Field(description="The memory index/key")
    score: int = Field(description="Relevance score from 1-5", ge=1, le=3)


class MemoryScoreResponse(BaseModel):
    """Wrapper around the full list of per-memory scores."""
    scores: List[MemoryScore] = Field(description="List of memory scores")


PROMPT_MEMORY_FILTER_OLD = """
You are a Memory Refiner for a retrieval-augmented agent. 
Given a query and retrieved memory items:

1. Keep only the memories directly useful to the query.
2. Discard irrelevant, outdated, or redundant items.
3. Merge the useful items into one short, clear memory text.
   - Deduplicated
   - Keep all the factual details.

Instructions:
- Output only the final refined memory text.
- If nothing is relevant, output an empty string.
- Do not explain or return lists/JSON.

Query: {query}

Retrieved Memories:
{original_memories}

Final Refined Memory:

"""

PROMPT_MEMORY_FILTER = """You are an expert Memory Refiner for a retrieval-augmented agent. Your task is to evaluate the relevance of retrieved memories in relation to a user query.

# TASK: 
Given a user query and a list of retrieved memories in the format of [memory_index]: memory_value, rate the relevance of each memory to the query on a scale from 1 to 3.
The scores will be used to filter out irrelevant, unhelpful, or outdated memories.
Then, return a JSON object containing the scores for each memory.

# GUIDELINES:
1. Scoring Criteria:
    - Score 3: The memory is very relevant and directly helps in answering the query.
    - Score 2: The memory might be useful or somewhat relevant to the query. It could provide some context or background information. It might not be directly necessary but still has value.
    - Score 1: The memory is completely unrelated, unhelpful or outdated for answering the query. It does not contribute any useful information to answering the query.

2. Evaluation Considerations:
    - Focus on the relevance of the memory content to the specific query.
    - Ensure that each memory is evaluated independently based on its own content.
    - Be objective and consistent in your scoring.

# OUTPUT FORMAT:
Return a JSON object with a "scores" array. Each entry should have:
- "index": the exact memory index (e.g., "Mike's birthday")
- "score": relevance score from 1 to 3

Example output:
{{
    "scores": [
        {{"index": "Mike's birthday", "score": 3}},
        {{"index": "Stacy's favorite color", "score": 1}},
        {{"index": "Mike's family gathering", "score": 2}}
    ]
}}


User Query: {query}

Retrieved Memories:
{memories_text}

Evaluate all memories and provide a score for each one.

Output:
"""

class MemoryFilter:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model_client = ChatCompletionModel(cfg)

    def filter_memory(
        self,
        query: str,
        memory_results: List["MemoryEntry"],
    ) -> List["MemoryEntry"]:
        """
        Use an LLM to score retrieved memories against the current query
        and drop the ones deemed irrelevant.


        Args:
            query: The original query/context string
            memory_results: List of retrieved MemoryEntry objects to filter

        Returns:
            Filtered list of MemoryEntry objects
        """
        if not memory_results:
            return memory_results

        # Build the per-memory listing using the [memory_index]: value template the LLM expects.
        memories_text = "\n".join(
            f"[{entry.index}]: {entry.get_memory_value()}"
            for entry in memory_results
        )

        # Bundle prompt arguments for the LLM call.
        prompt_args = {
            "query": query,
            "memories_text": memories_text,
        }

        try:
            # Ask the LLM and parse via the structured response schema.
            response = self._model_client.invoke(
                input=PROMPT_MEMORY_FILTER,
                prompt_args=prompt_args,
                response_format=MemoryScoreResponse,
            )

            score_lookup = {item.index: item.score for item in response.scores}

            # Sanity check: the LLM should return a score for each memory we sent.
            if len(score_lookup) != len(memory_results):
                logger.warning(
                    f"LLM returned {len(score_lookup)} scores but expected {len(memory_results)}. "
                    f"Skipping filter."
                )
                return memory_results

            # Threshold + tag with (entry, llm_score, search_score) tuples for downstream sort.
            scored_results = []
            for entry in memory_results:
                llm_score = score_lookup.get(entry.index, 0)
                if llm_score >= 2:  # keep relevance levels 2 and 3
                    scored_results.append((entry, llm_score, entry.score))

            # Order primarily by LLM score, breaking ties with the search score.
            scored_results.sort(key=lambda triple: (triple[1], triple[2]), reverse=True)

            filtered_results = [entry for entry, _, _ in scored_results]

            logger.info(
                f"LLM filtering: kept {len(filtered_results)}/{len(memory_results)} memories "
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Error during LLM filtering: {e}. Returning all memories.")
            return memory_results
