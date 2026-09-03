from __future__ import annotations
from dataclasses import dataclass, field
from importlib import metadata
import logging
from typing import List, Optional, Union, Dict, Any
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.memory_entry import MemoryEntry
from megamem.utils.llm import ChatCompletionModel
from megamem.core.memory import AgentMemory

from megamem.builder.memory_builder import (
    PROMPT_BUILD_MEMORY,
    MemoryBuilder,
    MemoryOutputsWithTurns
)


# Initialize module logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic chat/Teams message data model (provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedChatMessage:
    """Provider-agnostic chat/Teams message representation.

    Parsers for specific formats (Microsoft Graph Teams, Slack, etc.) should
    convert their raw data into this structure before passing to add_chats.
    """
    message_id: str
    body_text: str = ""               # plain text (HTML already stripped)
    sender_name: str = ""
    sender_address: str = ""          # email address of sender
    sent_datetime: str = ""           # ISO 8601 format
    to_recipients: List[Dict[str, str]] = field(default_factory=list)  # [{"name": ..., "address": ...}]
    cc_recipients: List[Dict[str, str]] = field(default_factory=list)
    subject: str = ""
    conversation_id: str = ""         # thread grouping key
    thread_id: str = ""               # thread identifier
    topic: str = ""                   # thread/channel topic
    conversation_type: str = ""       # e.g. "meeting", "chat", "channel"
    thread_type: str = ""             # e.g. "meeting", "chat"
    importance: str = "Normal"
    has_attachments: bool = False
    mentions: List[Dict] = field(default_factory=list)
    file_data: List[Dict] = field(default_factory=list)
    links: List[str] = field(default_factory=list)


# LLM prompt for episodic memory generation
PROMPT_EPISODIC_MEMORY = """
You are an expert episodic memory generator that creates episodic memory summaries from conversation segments.

# TASK:
Your task is to generate an episodic memory with an index and a detailed summary based on the provided conversation segment.

Generate the episodic memory in the following format:
EpisodicIndex: [6-8 word summary that captures the main topic, entity or event of the episode]
EpisodicValue: [1-3 sentence of descriptive summary of the conversation]

# GUIDELINES:
1. The EpisodicIndex
- Create a short index (6-8 words) that captures the main topic or event of the episode.
- Always include the specific context (e.g., domain, or entity) from the source text in the Index to avoid vague terms.

2. The EpisodicValue
- Generate an episodic summary (1-3 sentences) that captures:
  - The main information of the conversation segment, including the main topic, theme, or event being discussed.
  - Relevant participants in the conversation, refer to the participants by their names if available.
  - Use the original wordings from the conversation when possible.
- Focus on "what happened" rather than specific details. The summary is meant to provide context for future retrieval.
- Make the summary self-contained and understandable without the original conversation.
- If images are present, consider the visual content as part of the textual context.
- Use only information present in the conversation segment to generate the summary; do not add external knowledge or infer beyond the content.
- If the conversation is between a user and an AI assistant, focus on the user's inputs and the overall context rather than the assistant's responses.

Input Conversation Segment:
{content}

Output:
"""


class EpisodicMemoryOutput(BaseModel):
    episodic_index: str = Field(
        description="A short 6-8 word summary that captures the main topic, entity or event of the episode"
    )
    episodic_value: str = Field(
        description="A detailed 1-4 sentence episodic summary providing context and narrative"
    )

class ChatMemoryBuilder(MemoryBuilder):

    def __init__(self, cfg: DictConfig, megamem: AgentMemory, model_client: ChatCompletionModel):
        super().__init__(cfg, megamem, model_client)

    def build_memory_entries(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
    ) -> List[MemoryEntry]:
        """Extract memory entries from a chat segment.

        Args:
            content: Normalized content (text or dict).

        Returns:
            Memory entries extracted from ``content``.

        Note:
            Cue indices are produced by the same LLM call as primary indices.
        """

        # Pull the indexed message list out of the normalized payload.
        segment_messages = content.get("segment_messages") if isinstance(content, dict) else None

        if not segment_messages:
            logger.warning("No segment_messages in content, cannot use turn-based extraction")
            return []

        # Resolve the timestamp string (strip any preceding "...on " prefix).
        ts = metadata.get("timestamp", "N/A") if metadata else "N/A"
        if " on " in ts:
            ts = ts.split(" on ")[-1]

        content_text = content.get("text", "") if isinstance(content, dict) else content

        # Run the turn-based extraction prompt.
        memories_with_turns = self._model_client.invoke(
            input=PROMPT_BUILD_MEMORY,
            prompt_args={
                "content": content_text,
                "timestamp": ts,
            },
            response_format=MemoryOutputsWithTurns,
        )

        memory_entries: List[MemoryEntry] = []
        for pos, mem_out in enumerate(memories_with_turns.entries):
            try:
                # Pull the actual conversation text for the requested turns.
                extracted_value = self._extract_text_from_turns(
                    segment_messages,
                    mem_out.turn_ranges,
                )

                if not extracted_value.strip():
                    logger.debug(f"  Memory {pos+1}: Empty extracted value, skipping")
                    continue

                logger.debug(f"  Memory {pos+1} extracted value (first 200 chars):\n    {extracted_value[:200]}...")

                # Format cue indices supplied by the LLM (validated/deduped first).
                cue_indices_str = ""
                if mem_out.cue_indices:
                    validated_cues = self._validate_cue_indices(
                        mem_out.cue_indices,
                        mem_out.index,
                    )
                    cue_indices_str = "||".join(validated_cues)

                # Forward any extra metadata keys not already consumed by MemoryEntry.
                _consumed_keys = {
                    "creation_time", "timestamp", "episodic_memory_id",
                    "segment_topic", "segment_index", "image_urls",
                }
                extra_md = {
                    k: v
                    for k, v in (metadata or {}).items()
                    if k not in _consumed_keys and v is not None
                }

                episodic_id = metadata.get("episodic_memory_id") if metadata else None
                entry = MemoryEntry(
                    memory_type="factual",
                    index=mem_out.index,
                    value=extracted_value,  # raw conversation snippet, not LLM-rewritten
                    creation_time=metadata["creation_time"],
                    timestamp=metadata.get("timestamp", ""),
                    cue_indices=cue_indices_str,
                    episodic_memory_ids=[episodic_id] if episodic_id else [],
                    extra_metadata=extra_md if extra_md else {},
                )
                memory_entries.append(entry)
            except Exception as exc:
                # Best-effort identifier when the entry itself failed to parse.
                idx_label = getattr(mem_out, 'index', f'memory_{pos+1}')
                logger.error(f"Failed to extract turns for '{idx_label}': {exc}")
                continue

        logger.debug(
            f"\n{'='*80}\nDEBUG: Successfully created {len(memory_entries)} memory entries "
            f"from {len(memories_with_turns.entries)} LLM outputs\n{'='*80}\n"
        )

        # Defensive: tag every entry as factual.
        for entry in memory_entries:
            entry.memory_type = "factual"

        return memory_entries
    
    def generate_episodic_memory(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
    ) -> Optional[MemoryEntry]:
        """Build a high-level episodic summary via the LLM.

        Args:
            content: Conversation payload (text or multimodal dict from
                :func:`normalize_content`).
            metadata: Extra metadata to attach to the resulting entry.

        Returns:
            A new episodic ``MemoryEntry`` or ``None`` if extraction failed.
        """
        try:
            # After normalize_content the payload is either a string or a
            # dict with a "text" field; pick the textual portion either way.
            content_text = content["text"] if isinstance(content, dict) and "text" in content else content

            episodic_output = self._model_client.invoke(
                input=PROMPT_EPISODIC_MEMORY,
                prompt_args={"content": content_text},
                response_format=EpisodicMemoryOutput,
            )

            md = metadata or {}
            episodic_entry = MemoryEntry(
                memory_type="episodic",
                index=f"[EPISODIC] {episodic_output.episodic_index}",
                value=episodic_output.episodic_value,
                creation_time=md.get("creation_time", ""),
                timestamp=md.get("timestamp", ""),
            )

            return episodic_entry

        except Exception as exc:
            logger.warning(f"Failed to generate episodic memory from segment: {exc}")
            return None

    def _extract_text_from_turns(
        self,
        segment_messages: List[Dict[str, Any]],
        turn_ranges: List,
    ) -> str:
        """Concatenate the text content of selected conversation turns.

        Args:
            segment_messages: Ordered message dicts (role + content).
            turn_ranges: TurnRange objects (1-based, inclusive).

        Returns:
            The combined text from the requested turns, joined by newlines.
        """
        extracted_parts: List[str] = []
        n_msgs = len(segment_messages)

        for tr in turn_ranges:
            # Convert 1-based inclusive bounds to 0-based list indices.
            start = tr.start_turn - 1
            end = tr.end_turn - 1

            if start < 0 or end >= n_msgs:
                logger.warning(
                    f"Turn range [{tr.start_turn}, {tr.end_turn}] out of bounds "
                    f"for {n_msgs} messages, skipping"
                )
                continue

            if start > end:
                logger.warning(
                    f"Invalid turn range: start ({tr.start_turn}) > end ({tr.end_turn}), swapping"
                )
                start, end = end, start

            # Walk every selected turn (indices already 0-based here).
            for pos in range(start, end + 1):
                msg = segment_messages[pos]
                msg_content = msg.get("content", "")

                # Multimodal turns arrive as a list of typed parts; flatten the text.
                if isinstance(msg_content, list):
                    text_chunks = [
                        part.get("text", "")
                        for part in msg_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    text_content = " ".join(text_chunks)
                else:
                    text_content = msg_content

                # The speaker name is already embedded by add.py, so no role prefix.
                extracted_parts.append(text_content)

        return "\n".join(extracted_parts)

    def _validate_cue_indices(
        self,
        cue_indices: List[str],
        primary_index: str,
    ) -> List[str]:
        """Clean LLM-generated cue indices and cap the list at 3.

        Args:
            cue_indices: Raw cue indices coming from the LLM.
            primary_index: Primary memory index used for self-overlap checks.

        Returns:
            The deduped, validated cue indices.
        """
        validated: List[str] = []
        seen: set = set()
        primary_lower = primary_index.lower()

        for raw in cue_indices:
            cue = raw.strip()

            if not cue:
                continue

            cue_lower = cue.lower()

            # Drop cues that duplicate something we already kept.
            if cue_lower in seen:
                logger.debug(f"Skipping duplicate cue index: '{cue}'")
                continue

            # Drop cues that simply restate the primary index.
            if cue_lower == primary_lower:
                logger.debug(f"Skipping cue index that matches primary index: '{cue}'")
                continue

            # Drop overly short cues (need at least 2 words for context).
            if len(cue.split()) < 2:
                logger.debug(f"Skipping single-word cue index: '{cue}'")
                continue

            seen.add(cue_lower)
            validated.append(cue)

        # Per guidelines, keep at most three cue indices per memory.
        return validated[:3]
