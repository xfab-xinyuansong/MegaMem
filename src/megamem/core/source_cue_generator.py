import logging
from typing import Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


SOURCE_TYPE_METADATA_KEYS: Dict[str, List[str]] = {
    "mail": ["sender", "subject", "recipients", "date"],
    "doc":  ["author", "title", "date"],
    "teams": ["participants", "topic", "conversation_type", "date"],
}

_DEFAULT_METADATA_KEYS: List[str] = ["author", "title", "date"]


def get_metadata_keys_for_type(data_type: str) -> List[str]:
    return SOURCE_TYPE_METADATA_KEYS.get(data_type, _DEFAULT_METADATA_KEYS)


PROMPT_SOURCE_CUE = """You are a source-description assistant for a memory system.

# TASK
Given metadata about a source, produce a single **1-2 sentence natural-language
description** that captures WHO, WHAT, and WHEN.

This description will be embedded as a vector and used for semantic search, so it
must contain enough detail for a user query like "that email from Sarah about the
budget" to match it with high similarity.

# GUIDELINES
1. Always include the source type ("Email", "Document", etc.).
2. Use ONLY the provided metadata fields — do not invent or assume missing details.
3. For emails: include sender name (not just email address), subject/topic, and date.
4. For documents: include title, author (if known), and date.
5. Write in natural, flowing English — NOT bullet points or JSON.
6. Keep it concise: 1-2 sentences, roughly 15-30 words.
7. Convert email addresses to human-readable names where possible
   (e.g., "sarah.johnson@contoso.com" → "Sarah Johnson").

# EXAMPLES

Input:
  data_type: mail
  sender: sarah.johnson@contoso.com
  subject: Q3 Budget Review
  date: 2026-02-10

Output:
  Email from Sarah Johnson about Q3 budget review, sent February 10 2026.

Input:
  data_type: doc
  title: Project Nexus - Q1 Status Report
  author: Jane Smith
  date: 2026-01-28

Output:
  Document by Jane Smith titled "Project Nexus - Q1 Status Report" from January 28 2026.

Input:
  data_type: mail
  sender: bob@example.com
  subject: Weekly Standup Notes
  date: 2026-02-14

Output:
  Email from Bob regarding weekly standup notes, sent February 14 2026.

Input:
  data_type: mail
  sender: alice@example.com
  subject: Re: Project Phoenix Status
  recipients: team-leads@example.com
  date: 2026-03-01

Output:
  Email from Alice about Project Phoenix status update to team leads, sent March 1 2026.

Input:
  data_type: doc
  title: Annual Performance Review Template
  date: 2026-01-15

Output:
  Document titled "Annual Performance Review Template" from January 15 2026.

Input:
  data_type: chat
  participants: Alice, Bob, Carol
  topic: Sprint Planning
  conversation_type: channel
  date: 2026-02-20

Output:
  Teams channel conversation about Sprint Planning with Alice, Bob, and Carol from February 20 2026.

Input:
  data_type: chat
  participants: Dave, Eve
  conversation_type: meeting
  date: 2026-03-01

Output:
  Teams meeting chat between Dave and Eve from March 1 2026.

# INPUT
{source_metadata}

# OUTPUT
Produce the source description (1-2 sentences, no JSON):
"""


class SourceCueDescription(BaseModel):
    description: str = Field(
        description="A 1-2 sentence natural-language description of the source"
    )


class SourceCueGenerator:

    def __init__(self, cfg: DictConfig, model_client: Optional[ChatCompletionModel] = None):
        self.cfg = cfg
        self._model_client = model_client or ChatCompletionModel(cfg)

    def generate_source_cue(self, source_metadata: Dict[str, str]) -> str:
        data_type = source_metadata.get("data_type", "")

        allowed_keys = ["data_type"] + get_metadata_keys_for_type(data_type)
        filtered_metadata = {
            field_key: field_value
            for field_key, field_value in source_metadata.items()
            if field_key in allowed_keys and field_value
        }

        metadata_text = "\n".join(
            f"  {key}: {value}" for key, value in filtered_metadata.items()
        )

        prompt_args = {"source_metadata": metadata_text}

        try:
            result: SourceCueDescription = self._model_client.invoke(
                input=PROMPT_SOURCE_CUE,
                prompt_args=prompt_args,
                response_format=SourceCueDescription,
            )
            description = result.description.strip()
            logger.info(f"Generated source cue: {description[:80]}...")
            return description

        except Exception as e:
            logger.warning(f"Source cue generation failed: {e}. Using fallback.")
            return self._fallback_description(source_metadata)

    def _fallback_description(self, source_metadata: Dict[str, str]) -> str:
        data_type = source_metadata.get("data_type", "source")
        date = source_metadata.get("date", "")

        if data_type == "mail":
            sender = source_metadata.get("sender", "unknown sender")
            subject = source_metadata.get("subject", "no subject")
            date_part = f", sent {date}" if date else ""
            return f"Email from {sender} about {subject}{date_part}."
        elif data_type == "doc":
            title = source_metadata.get("title", "untitled document")
            author = source_metadata.get("author", "")
            author_part = f" by {author}" if author else ""
            date_part = f" from {date}" if date else ""
            return f"Document{author_part} titled \"{title}\"{date_part}."
        else:
            return f"Source ({data_type}) from {date}." if date else f"Source ({data_type})."
