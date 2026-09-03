from __future__ import annotations
from typing import List, Optional

from megamem.core.memory_entry import MemoryEntry
from typing import Dict, List, Optional
from omegaconf import DictConfig
from megamem.core.memory_entry import MemoryEntry
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.memory import (
    convert_memory_output,
)
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from megamem.core.memory import AgentMemory
import logging

from pydantic import BaseModel, Field

from megamem.builder.memory_builder import MemoryBuilder, MemoryOutputs

logger = logging.getLogger(__name__)


# Prompt for Document-level episodic memory (a.k.a. Document Summary, per plan.md §4
# mapping "Conversation -> Document, Episodic Memory -> Document Summary").
PROMPT_BUILD_DOCUMENT_EPISODIC = """
You are an expert document summarization assistant. Given a document segment, generate a
concise high-level summary that captures the topic, scope, and main thrust of the segment.

Produce the summary in the following format:
EpisodicIndex: 6-8 word title that captures the main topic, domain, or document section.
EpisodicValue: 1-3 sentences describing what the segment is about (topic, scope, main claims).

Guidelines:
- The EpisodicIndex must be self-contained (include domain, document type, or section qualifier).
- Do NOT include facts that belong to factual memories (those are extracted separately).
- Focus on "what this section is about" rather than enumerating low-level details.

Input Document Segment:
{content}

Output:
"""


class DocumentEpisodicMemoryOutput(BaseModel):
    episodic_index: str = Field(
        description="A short 6-8 word summary capturing the topic/scope of the document segment"
    )
    episodic_value: str = Field(
        description="A 1-3 sentence summary describing the document segment"
    )


PROMPT_BUILD_DOCUMENT_MEMORY = """
You are an expert assistant for knowledge extraction and memory construction. 
Your task is to extract both factual and procedural memories from a given document segment 
and represent them as memory entries for a long-term memory database.

=== Output Format ===
Each extracted memory should follow one of these formats:

Factual Memory:
MemType: Factual
MemIndex: <short, semantically rich title summarizing the key concept>
MemValue: <detailed factual content; preserve all informative details>

Procedural Memory:
MemType: Procedural
MemIndex: <concise, descriptive title of the process or workflow>
MemSteps:
1. <first step>
2. <second step>
3. ...
Summary: <brief description of the procedure’s purpose, outcome, or trigger condition>

=== Guidelines ===
1. Fidelity & Granularity
   - Preserve as much detail as possible from the segment.
   - Use the entire segment as one memory value when coherent.
   - Split only if the text clearly contains distinct, independent ideas or processes.
   - Never over-summarize; keep full definitions, examples, and conditions.

2. Detecting Procedural Memories
   - Classify as procedural if the text describes ordered steps, actions, or decision logic.
   - Common cues: “first”, “then”, “follow”, “process”, “step”, “if/when”, “procedure”.
   - Represent each action as a numbered MemStep.
   - Provide a Summary explaining what the procedure accomplishes.

3. MemIndex Construction
   - 3–8 words, compact but meaningful.
   - Include contextual or hierarchical qualifiers if needed (e.g., “System Recovery > Validation Step”).
   - Avoid vague titles like “Overview” or “Details” without context.

4. Factual Memory Writing
   - Use neutral, factual sentences.
   - Include definitions, metrics, examples, or role responsibilities if available.
   - Avoid meta phrases like “This section describes…”.

5. Procedural Memory Writing
   - List steps in order.
   - Make each step actionable and clear.
   - Include conditional logic if mentioned.
   - Keep Summary concise but descriptive.

6. Context Awareness
   - If the document has hierarchy (e.g., Document > Section > Subsection),
     reflect that context implicitly in the MemIndex.
   - Example: “Incident Management > Stage 2 - Assessment” 
     → MemIndex: “Incident Response - Stage 2 Assessment Process”.

=== Example ===
Input:
Incident Escalation Process
1. Detect potential outage and validate telemetry.
2. Notify incident commander.
3. Open communication bridge.
4. Escalate to leadership if impact persists beyond 30 minutes.

Output:
MemType: Procedural
MemIndex: Incident Escalation Process
MemSteps:
1. Detect potential outage and validate telemetry.
2. Notify the incident commander.
3. Open a communication bridge.
4. Escalate to leadership if the impact persists beyond 30 minutes.
Summary: A four-step escalation workflow ensuring timely leadership engagement when impact persists.

=== Final Instruction ===
Process the following segment:
{segment_content}

Generate the appropriate Factual or Procedural memory entries 
following the above format and guidelines.

"""

class DocumentMemoryBuilder(MemoryBuilder):

    def __init__(self, cfg: DictConfig, megamem: AgentMemory, model_client: ChatCompletionModel):
        super().__init__(cfg, megamem, model_client)
            

    def build_memory_entries(
        self, content: Union[str, Dict], metadata: Optional[Dict]
    ) -> List[MemoryEntry]:
        """Build memory entries from a document segment.

        Args:
            content: Text (or multimodal dict) the LLM will analyze.
            metadata: Extra metadata recorded alongside each entry.

        Returns:
            Memory entries derived from ``content``.
        """

        # Always rely on the document-specific extraction prompt.
        build_memory_prompt = PROMPT_BUILD_DOCUMENT_MEMORY
        response_format = MemoryOutputs

        # First try the multimodal path (text + images dict).
        memories = self.handle_multimodal_content(content, metadata, build_memory_prompt, response_format)

        if memories is None:
            # Fallback: pure text content.
            memories = self._model_client.invoke(
                input=build_memory_prompt,
                prompt_args={
                    "segment_content": content,
                },
                response_format=response_format,
            )

        # Convert raw LLM output into MemoryEntry objects (cues come later).
        memory_entries = convert_memory_output(memories, metadata, enable_cue_index=False)

        # Optionally enrich each entry with cue indices via a single batch call.
        if self.cfg.memory.enable_cue_index and memory_entries:
            try:
                memories_batch = [
                    {"index": e.index, "value": e.value}
                    for e in memory_entries
                ]

                cue_indices_map = self.cue_index_generator.generate_cue_indices_batch(memories_batch)

                for e in memory_entries:
                    cues = cue_indices_map.get(e.index, [])
                    e.cue_indices = "||".join(cues) if cues else ""

            except Exception as exc:
                logger.warning(f"Failed to generate cue indices in batch: {exc}")
                # Default to empty cues on any failure.
                for e in memory_entries:
                    e.cue_indices = ""

        return memory_entries

    def generate_episodic_memory(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
    ) -> Optional[MemoryEntry]:
        """Build a Document-level episodic summary via the LLM.

        plan.md §4 maps "Conversation -> Document" and "Episodic Memory -> Document
        Summary". This implementation mirrors EmailMemoryBuilder.generate_episodic_memory
        but uses a document-summary prompt.

        Args:
            content: Document segment payload (text or normalized multimodal dict).
            metadata: Extra metadata to attach to the resulting episodic entry.

        Returns:
            A new episodic ``MemoryEntry`` or ``None`` if extraction failed.
        """
        try:
            # After normalize_content the payload is either a string or a dict with
            # a "text" field; pick the textual portion either way.
            content_text = (
                content["text"] if isinstance(content, dict) and "text" in content else content
            )

            episodic_output = self._model_client.invoke(
                input=PROMPT_BUILD_DOCUMENT_EPISODIC,
                prompt_args={"content": content_text},
                response_format=DocumentEpisodicMemoryOutput,
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
            logger.warning(f"Failed to generate document episodic memory: {exc}")
            return None
