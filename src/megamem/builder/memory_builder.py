from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.general_api import GeneralBadRequestError
from megamem.core.cue_index_generator import CueIndexGenerator
from megamem.core.memory import AgentMemory, QueryMode
from megamem.core.memory_entry import MemoryEntry
from megamem.core.predictive_cue_generator import PredictiveCueGenerator
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.log import log_memory_building
from megamem.utils.memory import (
    MemoryUpdateDecision,
    generate_history,
    generate_metadata,
)
from megamem.utils.misc import get_current_timestamp, normalize_content

logger = logging.getLogger(__name__)

# Efficient prompt for turn-based memory extraction
PROMPT_BUILD_MEMORY = """
You are an expert memory extraction assistant. Your goal is to extract memories from a conversation segment by identifying important information that could be useful for future interactions.

# TASK:
The input is a conversation segment where every turn is indexed (1-based). Read the input conversation carefully, and extract ALL memories that could be useful for future reference.
Produce each memory as a structured entry in the following format:

Primary Index: A concise 6-8 word phrase summarizing the memory
Turn Ranges: Turn ranges (start_turn and end_turn indices) from the conversation that correspond to the memory content
Cue Indices: 1-4 short phrases to enhance retrieval by capturing different aspects of the memory content


# GUIDELINES:

1. Content and Scope:
    - A memory entry is a piece of information that could be useful for answering future questions or providing context in future interactions. It can be about facts, experiences, events, intentions, hobbies, preferences, beliefs, goals, or plans.
    - A memory can span multiple turns in the conversation.
    - Capture ALL memories that could be useful for future retrieval.
    - Do not include greetings, small talk, or filler.
    - Try to split distinct pieces of information into separate memories. Avoid having memory that spans more than 6 turns, as it may be too broad and less accurate for retrieval. If a memory spans more than 6 turns, try to split it into multiple memories with more specific turn ranges.
    - Range exclusivity: Do not create multiple memory entries that cover the exact same turn range. In that case, use primary index and cue indices to capture different aspects of the same memory.
    - Overlap handling: Overlap is permitted only if the memories capture significantly different topics that happen to intersect. However, prefer splitting ranges if possible. Avoid having multiple memories that share a large portion of the same turns, as it may lead to confusion during retrieval. If two memories have significant overlap in turn ranges, consider merging them into one memory with a more comprehensive primary index and diverse cue indices to capture different facets of the content.

2. Primary Index Format:
- The Primary Index is the primary identifier for the memory. It should concisely capture the essence of the memory content.
- The Primary Index must be a short, human-readable phrase that is self-contained and unambiguous.
- Each Primary Index should be unique and specific enough to distinguish it from other memories.
- Always include the specific context (e.g., domain, or entity) from the source text in the Primary Index to avoid vague terms. For example, instead of "Vacation", use "Alice's Japan Vacation". Instead of "Mike's plans", use "Mike's summer plans to visit Europe".

3. Cue Indices:
- **Definition**: A cue index is a concise phrase (2-4 words) that anchors a specific topic to a memory. It takes the following structure: [Main Entity] + [Key Aspect].
    - The **Main Entity** is the primary person, domain, or object involved in the memory (the "Who" or "What").
    - The **Key Aspect** specifies the event, preference, action, state, or object associated with the entity.
    Examples of Main Entity + Key Aspect patterns:
        - [Person] + [Event/Activity] → "Jane hiking trip", "Mike vacation"
        - [Person] + [Hobby/Preference] → "Michael Jazz music", "Sophie vegan diet"
        - [Person] + [Condition/State] → "Emma career change", "Liam health problems"
        - [Person] + [Object/Relation] → "Alice research paper", "David guitar"
        - [Domain] + [Attribute/Artifact] → "Project Orion timeline", "Product X features"

- **Specificity**: Avoid generic single words like "summer", "happiness", or "project meeting". Every cue index must be contextually anchored to the main entity, event, or domain mentioned in the memory. For example, instead of "hiking," use "Sarah hiking." The key aspect should reflect a concrete topic rather than a vague concept. For example, use "Mike mental health problems" instead of "Mike feelings."
- **Atomicity**: Each cue index must represent a single, indivisible aspect. Do not overload a cue with timestamps, specific numbers, or multiple descriptors. For example, use "Mike birthday party" instead of "Mike birthday party 2023". Avoid overspecification that limits generalizability.
- **Distinct Facets**: A memory could have multiple cue indices, each focusing on a different aspect of the memory to provide diverse viewpoints. Ideally, cue indices of one memory should not overlap in meaning. Each index must target a completely different dimension of the memory. Avoid generating cue indices that are similar to each other for the same memory. For example, don't create both "Project Phoenix kickoff" and "Project Phoenix launch" for the same memory.
- **Uniqueness**: Do not repeat the primary memory index as a cue index.
- **Purpose**: Cue indices could help with recall and reasoning by providing additional semantic keys beyond the primary index. They serve to link related memories together based on shared themes.
- How to generate cue indices:
    - Examine the corresponding turn ranges to identify aspects of the memory content that stand out.
    - Create 1-4 cue indices that capture different facets of the memory, that are not already covered by the primary index.

4. Turn Selection:
- Identify which specific turns contain the information for each memory.
- A turn range specifies start_turn and end_turn (both inclusive, 1-based).
- If the information for a memory is spread across multiple turns, include all relevant turns in the TurnRanges.
- You can specify multiple non-contiguous turn ranges if a memory spans separated parts.
- Exclude assistant acknowledgments or filler turns that do not contain substantive information.

# Example:

[Turn 1] Alice: "I'm planning a vacation to Japan next month!"
[Turn 2] Bob: "That sounds amazing! Where in Japan are you going?"
[Turn 3] Alice: "I'll be visiting Tokyo and Kyoto. I want to see the cherry blossoms and try authentic sushi."
[Turn 4] Bob: "Nice!"
[Turn 5] Alice: "Also, I'm going with my friend Mike, who loves Japanese culture as much as I do."
[Turn 6] Bob: "That's great, I hope you have a fantastic trip. By the way, my kids are performing at a concert in the park this afternoon."
[Turn 7] Alice: "No worries, what kind of concert is it?"
[Turn 8] Bob: "It's a local music concert, my kids will be playing the guitar and drums. It's been a lot of work preparing for it, but I'm really proud of them."

Extracted Memory:
PrimaryMemIndex: Alice's travel plans to Japan next month
TurnRanges: [[1, 3], [5, 5]]
CueIndices: ["Alice Tokyo Kyoto", "Alice cherry blossoms", "Alice trying sushi"]

PrimaryMemIndex: Bob's kids performing at a concert in the park
TurnRanges: [[6, 8]]
CueIndices: ["Bob kids guitar drums", "Bob local music concert", "Bob proud parent"]

# Now, extract memories from the following conversation segment:

Input Conversation:
{content}

Output:
"""

# LLM prompt template for memory update decisions
PROMPT_MEMORY_UPDATE_DECISION = """
You are a memory management assistant. Given a new memory entry and similar existing entries, determine whether to update an existing entry or add a new one.

NEW MEMORY ENTRY:
Index: {new_index}
Value: {new_value}

EXISTING SIMILAR ENTRIES:
{candidates_info}

INSTRUCTIONS:
1. Analyze if the new entry should update any existing entry based on semantic similarity and content overlap
2. If update is needed, determine which candidate is best to update
3. Decide if the memory index should be updated to better reflect the combined information. The updated index should remain concise and human-readable, capturing the essence of the merged content.
4. Generate 1-3 cue indices for the resulting memory:
   - Each cue is a 2-4 word phrase: [Main Entity] + [Key Aspect]
   - Examples: "Jane hiking", "Mike birthday party", "Project Orion launch"
   - Cues should cover different aspects of the memory content to enhance retrieval diversity
   - Avoid generic single words; anchor to specific entities
   - Do not repeat the primary index
"""

# Template for formatting candidate information in LLM prompts
PROMPT_CANDIDATE_FORMAT = """
Candidate {index}:
- Similarity Score: {score:.3f}
- Index: {index_text}
- Value: {value}
- Creation Time: {creation_time}
"""

class EpisodicMemoryOutput(BaseModel):
    episodic_index: str = Field(
        description="A short 6-8 word summary that captures the main topic, entity or event of the episode"
    )
    episodic_value: str = Field(
        description="A detailed 1-4 sentence episodic summary providing context and narrative"
    )

class MemoryOutput(BaseModel):
    memory_type: str = Field(description="Type of memory: 'Factual' or 'Procedural'.")
    index: str = Field(
        description="Use a short, specific phrase that captures the fact clearly, with enough detail to support retrieval."
    )
    value: str = Field(
        description="A concise but complete factual statement, supported directly by the conversation."
    )

    def __str__(self) -> str:
        return f"[*{self.index}*] {self.value}"

    def __repr__(self) -> str:
        return str(self)


class MemoryOutputs(BaseModel):
    entries: list[MemoryOutput] = Field(
        description="memories extracted from the content that are factual and verifiable.",
    )


class TurnRange(BaseModel):
    """Represents a range of turns to include in a memory."""
    start_turn: int = Field(description="Starting turn index (1-based, inclusive)")
    end_turn: int = Field(description="Ending turn index (1-based, inclusive)")


class MemoryOutputWithTurns(BaseModel):
    """Memory extraction output with turn-based value specification."""
    memory_type: str = Field(description="Type of memory: 'Factual' or 'Procedural'.")
    index: str = Field(
        description="Use a short, specific phrase that captures the fact clearly"
    )
    turn_ranges: List[TurnRange] = Field(
        description="List of turn ranges to extract as the memory value (1-based indices)"
    )
    cue_indices: List[str] = Field(
        default_factory=list,
        description="List of 1-3 cue indices for enhanced retrieval. Each cue is a 2-4 word phrase following [Main Entity] + [Key Aspect] pattern. Examples: 'Jane hiking', 'Mike birthday party', 'Project Orion timeline'. Avoid generic single words and do not repeat the primary index."
    )


class MemoryOutputsWithTurns(BaseModel):
    """Container for multiple memory extractions with turn-based values."""
    entries: List[MemoryOutputWithTurns] = Field(
        description="Memories extracted from the content"
    )


class MemoryBuilder(ABC):
    """All builders accept either text or file_path (exactly one required)."""

    def __init__(self, cfg: DictConfig, megamem: AgentMemory, model_client: ChatCompletionModel):
        self.cfg = cfg
        # Backing LLM client used by all derived builders.
        self._model_client = model_client

        # Memory store handle for read/write operations.
        self.megamem = megamem

        # Multimodal flag (config-driven, defaults to True).
        self.multimodal_support = cfg.memory.get("multimodal_support", True)

        self.cue_index_generator = CueIndexGenerator(
            cfg=self.cfg, model_client=self._model_client
        )

        self.predictive_cue_generator = PredictiveCueGenerator(
            cfg=self.cfg, model_client=self._model_client
        )

        # Similarity threshold above which an existing entry is considered for update.
        self.UPDATE_SCORE_THRESHOLD = cfg.memory.update_score_threshold

    def build(
        self,
        content: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict] = None,
    ) -> List[MemoryEntry]:
        """Build (and dedupe) memory entries from a piece of context."""

        # Standardize the various accepted input shapes.
        content = normalize_content(content, multimodal_support=self.multimodal_support)

        log_memory_building(content, self.megamem.get_user_id())

        # Annotate metadata with creation_time and any embedded image URLs.
        metadata = generate_metadata(content, metadata)

        # Step 1: optional episodic memory generation + storage.
        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)
        episodic_entry = None

        if enable_episodic:
            episodic_entry = self.generate_episodic_memory(content, metadata)

            if episodic_entry is not None:
                self.megamem.add(episodic_entry)
                # Forward the episodic id so downstream factual entries can backlink.
                metadata["episodic_memory_id"] = episodic_entry.index

        # Step 2: pull structured factual entries out of the context.
        memory_entries = self.build_memory_entries(content, metadata)

        # Step 3: optional predictive (extrinsic) cue generation.
        if memory_entries and self.cfg.memory.get("enable_predictive_cues", False):
            self._generate_predictive_cues_batch(memory_entries)

        # Step 4: upsert each entry through the dedupe-aware path.
        for entry in memory_entries:
            self.upsert_memory_entry(entry=entry)

        return memory_entries

    @abstractmethod
    def build_memory_entries(
        self, content: str, metadata: Optional[Dict]
    ) -> List[MemoryEntry]:
        """   
        Abstract method to build memory entries from content.
        Must be implemented by subclasses.

        Cue indices are always generated in the same LLM call as primary indices.
        """
        ...

    @abstractmethod
    def generate_episodic_memory(
        self, content: str, metadata: Optional[Dict]
    ) -> Optional[MemoryEntry]:
        """
        Abstract method to generate episodic memory from content.
        Must be implemented by subclasses.
        Returns None if episodic memory generation is not supported or fails.
        """
        ...
                
    def _generate_predictive_cues_batch(self, entries: List[MemoryEntry]) -> None:
        """Generate predictive cues for ``entries`` concurrently."""
        worker_count = min(len(entries), 5)

        def _process(entry: MemoryEntry):
            cues = self.predictive_cue_generator.generate(
                entry.index, entry.value, entry.get_cue_indices()
            )
            if cues:
                entry.predictive_cue_indices = "||".join(cues)
                logger.info(
                    "PCA: '%s' -> %d predictive cues: %s",
                    entry.index, len(cues), cues,
                )

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_process, e): e for e in entries}
            for fut in as_completed(futures):
                err = fut.exception()
                if err is not None:
                    failed = futures[fut]
                    logger.warning(
                        "PCA failed for '%s': %s", failed.index, err
                    )

    def handle_multimodal_content(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
        build_memory_prompt: str,
        response_format: Any,
    ) -> Optional[MemoryOutputs]:

        # Bail out unless the caller really has multimodal payload to send.
        if not (isinstance(content, dict) and self.multimodal_support and "image" in content):
            return None

        # Format the textual prompt and combine it with the supplied images.
        formatted_prompt = build_memory_prompt.format(
            content=content["text"],
            timestamp=metadata.get("timestamp", "") if metadata else "",
        )

        # Build a single user message with text + image parts.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": formatted_prompt}] + content["image"],
            }
        ]

        # Attempt the multimodal call; on image access errors, return None
        # so callers can fall back to the text-only path.
        try:
            return self._model_client.invoke(
                input=messages,
                response_format=response_format,
            )
        except GeneralBadRequestError as exc:
            err_str = str(exc)
            if "403" in err_str or "can not be accessed" in err_str:
                logger.warning(
                    f"Image URL access failed (403 error): {exc}. Falling back to text-only mode."
                )
            return None

    def _query_update_candidates(self, entry: MemoryEntry) -> List[MemoryEntry]:
        """Locate existing memories that might be merged with ``entry``.

        Returns:
            Memory entries whose similarity score clears the configured
            update threshold.
        """
        # Filter: same user, different batch, only un-linked factual primaries.
        where = {
            "$and": [
                # Drop entries from the same write batch (matching creation_time).
                {"creation_time": {"$ne": entry.creation_time}},
                # Drop cue indices that already point at a linked memory.
                {"linked_memory": {"$eq": ""}},
                # Episodic memories are immutable, so only consider factual ones.
                {"memory_type": {"$eq": "factual"}},
            ]
        }

        # Vector-similarity lookup against the primary index space only.
        query_results: List[MemoryEntry] = self.megamem.query(
            entry.index,
            top_k=5,
            where=where,
            query_mode=QueryMode.PRIMARY_ONLY,
            enhance_query=False,
        )

        # Keep only the candidates that beat the similarity bar.
        threshold = self.UPDATE_SCORE_THRESHOLD
        return [cand for cand in query_results if cand.score >= threshold]

    def upsert_memory_entry(
        self,
        entry: MemoryEntry,
    ) -> str:
        """Add or merge a single memory entry with dedupe awareness."""

        # Short-circuit if a memory with this exact index already exists.
        existing_entry = self.megamem.get(entry.index)
        if existing_entry is not None:
            # Log lots of diagnostic context so the duplicate can be traced.
            logger.warning(
                f"Duplicate memory index detected during upsert: '{entry.index}'\n"
                f"Existing: is_primary={existing_entry.is_primary_index()}, "
                f"is_cue={existing_entry.is_cue_index()}, "
                f"creation_time={existing_entry.creation_time}, "
                f"timestamp={existing_entry.timestamp}\n"
                f"New: is_primary={entry.is_primary_index()}, "
                f"is_cue={entry.is_cue_index()}, "
                f"creation_time={entry.creation_time}, "
                f"timestamp={entry.timestamp}\n"
                f"Values match: {existing_entry.value == entry.value}\n"
                f"This may indicate duplicate extraction or a race condition."
            )

            print(f"[Existing] {existing_entry.index} -> {existing_entry.value}")
            print(f"[*****New] {entry.index} -> {entry.value}")

            return entry.index

        # Vector lookup for memories worth considering for an update.
        update_candidates = self._query_update_candidates(entry)

        if not update_candidates:
            # No close neighbours — add as a brand-new entry.
            self.megamem.add(entry)
            return entry.index

        # Candidates exist — let the LLM choose between update and add.
        update_decision = self._decide_memory_update(entry, update_candidates)
        if update_decision["should_update"]:
            return self.update_memory(entry, update_decision)

        # LLM declined to update — add as fresh entry.
        self.megamem.add(entry)
        return entry.index

    def merge_memory(
        self,
        existing_entry: MemoryEntry,
        new_entry: MemoryEntry,
    ) -> MemoryEntry:
        """Combine two memory entries into a single merged entry.

        Args:
            existing_entry: The pre-existing memory.
            new_entry: The freshly built memory to fold in.

        Returns:
            A new ``MemoryEntry`` representing the union of both inputs.
        """
        # Cross-type merges are explicitly disallowed.
        if existing_entry.memory_type != new_entry.memory_type:
            logger.warning(
                f"Attempted to merge different memory types (existing: {existing_entry.memory_type}, "
                f"new: {new_entry.memory_type}). Memories must be of the same type."
            )
            raise ValueError("Cannot merge memories of different types")

        merged_value = f"{existing_entry.value} {new_entry.value}"
        merged_cue_indices = list(
            set(existing_entry.cue_indices.split("||") + new_entry.cue_indices.split("||"))
        )

        # Combine episodic memory IDs from both inputs (deduped).
        merged_episodic_ids: list = []
        for src in (existing_entry.episodic_memory_ids, new_entry.episodic_memory_ids):
            if src:
                merged_episodic_ids.extend(src)
        merged_episodic_ids = list(set(merged_episodic_ids))

        # Combine image URLs from both inputs (deduped).
        merged_image_urls: list = []
        for src in (existing_entry.image_urls, new_entry.image_urls):
            if src:
                merged_image_urls.extend(src)
        merged_image_urls = list(set(merged_image_urls))

        merged_extra: dict = {}
        if existing_entry.extra_metadata:
            merged_extra.update(existing_entry.extra_metadata)
        if new_entry.extra_metadata:
            merged_extra.update(new_entry.extra_metadata)

        return MemoryEntry(
            memory_type=existing_entry.memory_type,
            index=existing_entry.index,
            value=merged_value,
            user_id=existing_entry.user_id,
            creation_time=existing_entry.creation_time,
            timestamp=new_entry.timestamp,
            cue_indices="||".join(merged_cue_indices),
            episodic_memory_ids=merged_episodic_ids,
            image_urls=merged_image_urls,
            extra_metadata=merged_extra if merged_extra else {},
        )

    def update_memory(
        self,
        entry: MemoryEntry,
        update_decision: Dict[str, Any],
    ) -> str:
        """Apply the LLM-driven update of an existing memory.

        Args:
            entry: The new memory entry that triggered the update.
            update_decision: Structured LLM decision (best candidate, new
                index, etc.).

        Returns:
            The index under which the merged entry was stored.
        """
        best_candidate: MemoryEntry = update_decision["best_candidate"]

        # Episodic memories are immutable — fall back to a plain add.
        if best_candidate.memory_type != "factual":
            logger.warning(
                f"Attempted to update non-factual memory (type: {best_candidate.memory_type}). "
                f"Only factual memories can be updated. Adding as new entry instead."
            )
            self.megamem.add(entry)
            return entry.index

        # Index produced by the LLM update decision.
        updated_index = update_decision["updated_index"]

        # Detect whether the existing value already carries timestamp headers
        # from prior updates; if so we just append, otherwise we wrap both.
        has_timestamp_headers = "\n[" in best_candidate.value and "]\n" in best_candidate.value
        if has_timestamp_headers:
            updated_value = f"{best_candidate.value}\n[{entry.timestamp}]\n{entry.value}"
        else:
            updated_value = (
                f"[{best_candidate.timestamp}]\n{best_candidate.value}\n"
                f"[{entry.timestamp}]\n{entry.value}"
            )

        # If renaming the index would collide with another existing entry,
        # disambiguate with a timestamped suffix.
        existing_entry = self.megamem.get(updated_index)
        if existing_entry and updated_index != best_candidate.index:
            updated_index = f"{updated_index}. (Added on {get_current_timestamp()})"

        # Snapshot source cues before deletion so provenance survives a rename or merge.
        retained_source_cues: List[MemoryEntry] = []
        for cue_index in best_candidate.get_cue_indices():
            cue_entry = self.megamem.get(cue_index)
            if (
                cue_entry
                and cue_entry.is_cue_index()
                and cue_entry.cue_type == "source"
            ):
                retained_source_cues.append(cue_entry)

        # Drop the soon-to-be-replaced original entry.
        self.megamem.delete(best_candidate.index)

        # Pull the cue indices that the LLM produced as part of this decision.
        updated_cue_indices = update_decision.get("updated_cue_indices", []) or []

        # Sanitize cue indices when we have any.
        if updated_cue_indices:
            validated_cues: list = []
            seen: set = set()
            updated_index_lower = updated_index.lower()
            for raw in updated_cue_indices:
                cue = raw.strip()
                if not cue:
                    continue
                cue_lower = cue.lower()
                if (
                    cue_lower in seen
                    or cue_lower == updated_index_lower
                    or len(cue.split()) < 2
                ):
                    continue
                seen.add(cue_lower)
                validated_cues.append(cue)
            updated_cue_indices = validated_cues[:3]

        # Track the lineage of this update for later inspection.
        history = generate_history(entry, best_candidate)

        # Merge image URLs from both inputs (deduped).
        updated_image_urls: list = []
        for src in (best_candidate.image_urls, entry.image_urls):
            if src:
                updated_image_urls.extend(src)
        updated_image_urls = list(set(updated_image_urls))

        # Merge episodic memory IDs from both inputs (deduped).
        updated_episodic_memory_ids: list = []
        for src in (best_candidate.episodic_memory_ids, entry.episodic_memory_ids):
            if src:
                updated_episodic_memory_ids.extend(src)
        updated_episodic_memory_ids = list(set(updated_episodic_memory_ids))

        # Carry forward extra_metadata (source_conv_idx, source_session, etc.)
        # from both entries; the new entry's values take precedence on conflict.
        merged_extra: dict = {}
        if best_candidate.extra_metadata:
            merged_extra.update(best_candidate.extra_metadata)
        if entry.extra_metadata:
            merged_extra.update(entry.extra_metadata)

        new_memory_entry = MemoryEntry(
            memory_type=best_candidate.memory_type,
            index=updated_index,
            value=updated_value,
            creation_time=entry.creation_time,
            timestamp=entry.timestamp,
            cue_indices="||".join(updated_cue_indices),
            history=history,
            image_urls=updated_image_urls,
            episodic_memory_ids=updated_episodic_memory_ids,
            extra_metadata=merged_extra if merged_extra else {},
        )
        self.megamem.add(new_memory_entry)

        # Restore source-cue links without reclassifying them as topical cues.
        for source_cue in retained_source_cues:
            if not source_cue.index:
                continue
            self.megamem.add_source_cue(
                source_description=source_cue.index,
                linked_memory_indices=[updated_index],
                data_type=source_cue.data_type or "",
                timestamp_unix=source_cue.timestamp_unix or 0,
                extra_metadata=dict(source_cue.extra_metadata or {}),
            )

        # Operational log line for the update.
        logger.info(
            "\n" + "-" * 60 + "\n"
            f"MEMORY STORE: Update|{entry.creation_time}|{self.megamem.get_user_id()}\n"
            f"Index: {best_candidate.index} -> {updated_index}\n"
            f"Value: {updated_value}\n" + "-" * 60
        )
        return updated_index

    def _decide_memory_update(self, new_entry, update_candidates) -> Dict[str, Any]:
        """Ask the LLM whether to merge ``new_entry`` with one of the candidates.

        Args:
            new_entry: The freshly built memory entry under consideration.
            update_candidates: Pre-filtered candidates from similarity search.

        Returns:
            A plain dict describing the LLM verdict and (optionally) the
            chosen candidate.
        """
        # Build a structured snapshot of every candidate.
        candidates_info = [
            {
                "index": pos,
                "score": cand.score,
                "index_text": cand.index,
                "value": cand.value,
                "creation_time": cand.creation_time,
            }
            for pos, cand in enumerate(update_candidates)
        ]

        prompt_args = {
            "new_index": new_entry.index,
            "new_value": new_entry.value,
            "candidates_info": self._format_candidates_for_prompt(candidates_info),
        }

        try:
            # Pydantic-typed response format keeps parsing deterministic.
            decision: MemoryUpdateDecision = self._model_client.invoke(
                input=PROMPT_MEMORY_UPDATE_DECISION,
                prompt_args=prompt_args,
                response_format=MemoryUpdateDecision,
            )

            decision_dict = decision.model_dump()
            if decision.should_update and decision.best_candidate_index is not None:
                decision_dict["best_candidate"] = update_candidates[decision.best_candidate_index]
            else:
                decision_dict["best_candidate"] = None

            return decision_dict

        except Exception as exc:
            # Defensive fallback when the LLM response cannot be parsed.
            return {
                "should_update": False,
                "best_candidate": None,
                "updated_value": None,
                "updated_index": new_entry.index,
                "reasoning": f"LLM parsing error: {exc}",
            }

    def _format_candidates_for_prompt(self, candidates_info: List[Dict]) -> str:
        """Render candidate memory dicts into a single prompt-friendly string."""
        formatted = [
            PROMPT_CANDIDATE_FORMAT.format(
                index=candidate["index"],
                score=candidate["score"],
                index_text=candidate["index_text"],
                value=candidate["value"],
                creation_time=candidate["creation_time"],
            )
            for candidate in candidates_info
        ]
        return "\n".join(formatted)
