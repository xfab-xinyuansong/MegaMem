from __future__ import annotations

import logging
from typing import Dict, List

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


PROMPT_CUE_GENERATION = """You are a memory-indexing assistant optimized for knowledge retrieval. Your goal is to create "Cue Indices" that serve as semantic anchors for specific memories.

# TASK
For each memory provided, generate 1-3 short, meaningful CUE INDICES that can later help recall or reason about that memory. Provide the cue indices as a list of strings for each memory.

# GUIDELINES
1. **Definition**: A cue index is a concise phrase (2-4 words) that anchors a specific topic to a memory. It uses the following structure: [Main Entity] + [Key Aspect].
    - The **Main Entity** is the primary person, domain, or object involved in the memory (the "Who" or "What").
    - The **Key Aspect** specifies the event, preference, action, state, or object associated with the entity.
    Examples of Main Entity + Key Aspect patterns:
        - [Person] + [Event/Activity] → "Jane hiking trip", "Mike vacation"
        - [Person] + [Hobby/Preference] → "Michael Jazz music", "Sophie vegan diet"
        - [Person] + [Condition/State] → "Emma career change", "Liam health problems"
        - [Person] + [Object/Relation] → "Alice research paper", "David guitar"
        - [Domain] + [Attribute/Artifact] → "Project Orion timeline", "Product X features"

2. **Specificity**: Avoid generic single words like "summer", "happiness", or "project meeting". Every cue index must be contextually anchored to the main entity, event, or domain mentioned in the memory. For example, instead of "hiking," use "Sarah hiking." The key aspect should reflect a concrete topic rather than a vague concept. For example, use "Mike mental health problems" instead of "Mike feelings."
3. **Atomicity**: Each cue index must represent a single, indivisible aspect. Do not overload a cue with timestamps, specific numbers, or multiple descriptors. For example, use "Mike birthday party" instead of "Mike birthday party 2023". Avoid overspecification that limits generalizability.
4. **Distinct Facets**: A memory could have multiple cue indices, each focusing on a different aspect of the memory to provide diverse viewpoints. Ideally, cue indices of one memory should not overlap in meaning. Each index must target a completely different dimension of the memory. Avoid generating cue indices that are similar to each other for the same memory. For example, don't create both "Project Phoenix kickoff" and "Project Phoenix launch" for the same memory.
5. **Uniqueness**: Do not repeat the primary memory index as a cue index.
6. **Purpose**: Cue indices could help with recall and reasoning by providing additional semantic keys beyond the primary index. They serve to link related memories together based on shared themes.


# EXAMPLES
Primary Index: "Jane's hiking trip to Appalachian Trail"
Memory Value: "Last summer, Jane went on a week-long hiking trip along the Appalachian Trail. She enjoyed the scenic views and challenging trails."
Cue indices: ["Jane hiking","Appalachian Trail views","Jane summer trip"]

Primary Index: "Mike's surprise birthday party"
Memory Value: "Mike's friends organized a surprise birthday party for him at his favorite restaurant Bistro Max."
Cue indices: ["Mike birthday party", "Mike favorite restaurant", "Mike friends gathering"]

Primary Index: "Project Orion launch delay"
Memory Value: "The launch of Project Orion has been delayed due to unforeseen technical issues that need to be resolved."
Cue indices: ["Project Orion launch", "Project Orion technical issues"]

Primary Index: "Emma went swimming"
Memory Value: "Emma went swimming during her vacation".
Cue indices: ["Emma swimming"]

# MEMORIES TO PROCESS
{memories}

"""


class MemoryCueIndices(BaseModel):
    memory_index: str = Field(description="The primary memory index")
    cue_indices: List[str] = Field(
        description="List of cue indices generated for this memory"
    )


class BatchCueIndices(BaseModel):
    results: List[MemoryCueIndices] = Field(
        description="List of cue indices for each memory"
    )


class CueIndexGenerator:
    """Backward-compatible cue index generator for memories.

    New builders generate cue and primary indices in one model call. This class remains
    available for callers using the earlier two-stage workflow.
    """

    def __init__(self, cfg: DictConfig, model_client: ChatCompletionModel):
        self.cfg = cfg
        self._model_client = model_client

    def generate_cue_indices_batch(
        self,
        memories: List[Dict[str, str]],
    ) -> Dict[str, List[str]]:
        """
        Produce cue indices for several memories in a single LLM round-trip.

        Args:
            memories: List of dictionaries with 'index' and 'value' keys

        Returns:
            Dictionary mapping memory indices to their cue indices
        """
        logger.warning(
            "CueIndexGenerator.generate_cue_indices_batch() is deprecated. "
            "Cue indices are now generated together with memory extraction."
        )

        # Compose the bullet-list of memories that the prompt expects.
        chunks = []
        for pos, mem in enumerate(memories, 1):
            chunks.append(f"\nMemory {pos}:\nPrimary Index: {mem['index']}\nMemory Value: {mem['value']}")
        memories_text = "".join(chunks)

        prompt_args = {
            "memories": memories_text,
        }

        try:
            result: BatchCueIndices = self._model_client.invoke(
                input=PROMPT_CUE_GENERATION,
                prompt_args=prompt_args,
                response_format=BatchCueIndices,
            )

        except Exception:
            logger.warning("Cue index generation failed; returning empty cue indices.")
            result = BatchCueIndices(
                results=[
                    MemoryCueIndices(
                        memory_index=mem["index"],
                        cue_indices=[],
                    )
                    for mem in memories
                ]
            )

        # Flatten the structured response into a {primary_index: [cue,...]} mapping.
        return {item.memory_index: item.cue_indices for item in result.results}

    def generate_cue_indices(
        self,
        memory_value: str,
        primary_index: str,
    ) -> List[str]:
        """
        Produce cue indices for a single memory.

        Args:
            memory_value: The memory content
            primary_index: The primary memory index

        Returns:
            List of cue indices
        """
        # Reuse the batch helper with a one-element list.
        result = self.generate_cue_indices_batch(
            [{"index": primary_index, "value": memory_value}]
        )

        # Look up cues by primary index, defaulting to an empty list.
        return result.get(primary_index, [])
