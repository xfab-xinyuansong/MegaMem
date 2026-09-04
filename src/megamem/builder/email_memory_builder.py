"""
Email Memory Builder

Extracts factual and episodic memories from email messages.
Accepts a generic email format (not tied to any specific provider).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.builder.memory_builder import MemoryBuilder
from megamem.core.memory import AgentMemory
from megamem.core.memory_entry import MemoryEntry
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.log import log_memory_building
from megamem.utils.misc import get_current_timestamp

logger = logging.getLogger(__name__)


@dataclass
class NormalizedEmail:
    """Provider-agnostic email representation.

    Parsers for specific formats (Microsoft Graph, Gmail, etc.) should convert
    their raw data into this structure before passing to EmailMemoryBuilder.
    """
    subject: str
    sender_name: str
    sender_address: str
    to_recipients: List[Dict[str, str]] = field(default_factory=list)  # [{"name": ..., "address": ...}]
    cc_recipients: List[Dict[str, str]] = field(default_factory=list)
    sent_datetime: str = ""          # ISO 8601 format
    body_text: str = ""              # plain text (HTML already stripped)
    conversation_id: str = ""        # thread grouping key
    message_id: str = ""             # unique identifier


class EmailMemoryOutput(BaseModel):
    """Single factual memory extracted from an email."""
    index: str = Field(
        description="A concise 6-8 word phrase summarizing the memory"
    )
    value: str = Field(
        description="A detailed factual statement capturing the information from the email"
    )
    cue_indices: List[str] = Field(
        default_factory=list,
        description=(
            "1-3 cue indices for enhanced retrieval. Each cue is a 2-4 word phrase "
            "following [Main Entity] + [Key Aspect] pattern."
        ),
    )


class EmailMemoryOutputs(BaseModel):
    """Container for multiple factual memories extracted from an email."""
    entries: List[EmailMemoryOutput] = Field(
        description="Factual memories extracted from the email"
    )


class EmailFilterOutput(BaseModel):
    """LLM decision on whether an email thread has enough substance to extract memories."""
    has_extractable_content: bool = Field(
        description="True if the email thread contains meaningful information worth remembering"
    )


class EmailEpisodicMemoryOutput(BaseModel):
    """Episodic memory summarizing an email."""
    episodic_index: str = Field(
        description="A short 6-8 word summary capturing the main topic of the email"
    )
    episodic_value: str = Field(
        description="A 1-3 sentence summary of the email's overall content and purpose"
    )


PROMPT_EMAIL_THREAD_FILTER = """
You are an email triage assistant. Decide whether an email thread contains enough meaningful information to extract lasting memories from.

# THREAD SUMMARY:
- Subject: {subject}
- Participants: {participants}
- Date range: {date_range}
- Number of emails: {num_emails}

# EMAILS IN THREAD:
{thread_content}

# GUIDELINES:
Mark the thread as NOT having extractable content if the entire thread consists of:
- Simple acknowledgements or pleasantries ("Thanks!", "Sounds good", "Welcome aboard!")
- Auto-replies or out-of-office messages
- Meeting acceptance/decline/tentative notifications
- System-generated messages (calendar updates, automated alerts)
- Welcome/onboarding email chains with only generic content
- Short replies that only confirm receipt without adding new information

Mark the thread as having extractable content if emails contain substantive information.

Respond with only a verdict: has_extractable_content true or false.
true means the thread contains meaningful information worth remembering; false means it does not.
"""

PROMPT_BUILD_EMAIL_MEMORY = """
You are an expert memory extraction assistant. Your goal is to extract factual memories from an email message that could be useful for future reference.

# TASK:
Read the email below and extract ALL factual memories that could be useful for future retrieval.
Produce each memory as a structured entry in the following format:

Primary Index: A concise 6-8 word phrase summarizing the memory
Value: A detailed factual description capturing the information from the email
Cue Indices: 1-4 short phrases to enhance retrieval by capturing different aspects of the memory content

# GUIDELINES:

1. Content and Scope:
    - A memory entry is a piece of information that could be useful for answering future questions or providing context in future interactions.
    - Focus on the substantive content of the email: facts, decisions, action items, commitments, events, and notable information.
    - Do not include email signatures, greetings, boilerplate, or auto-generated content. Statements such as "Let's catch up soon" or "Thanks for your help" should not be extracted as memories.
    - Split distinct pieces of information into separate memories.
    - When dates and times are mentioned, replace relative references (e.g., "tomorrow", "next week") with absolute dates based on the email sent date.
    - Use only information present in the email to generate the memory; do not add external knowledge or infer beyond the content.

2. Primary Index:
    - The primary index should be a short, human-readable phrase that is self-contained and unambiguous.
    - The primary index should always include specific context (person, project, topic) to avoid vague terms. For example, instead of "Meeting update", use "Alice's Q1 roadmap meeting update". Instead of "Project deadline", use "Project Atlas final report deadline".
    - Even if extracting multiple memories from the same email, each memory should have a distinct and contextualized primary index. Do not omit the main entity or topic from the index.

3. Memory Value:
    - Should be 1-3 sentences that clearly and concisely capture the factual information from the email.
    - Use neutral, factual wording. Use original phrasing from the email when possible.
    - Replace pronouns with specific names or entities for clarity.
    - When dates and times are mentioned, convert relative references to absolute dates based on the email's sent date.

4. Cue Indices:
    - A cue index is a concise phrase (2-4 words) that anchors a specific topic to a memory.
    - Each cue is a 2-4 word phrase: [Main Entity] + [Key Aspect].
    - Cover different facets of the memory content.
    - Do not repeat the primary index.
    - Examples: "Atlas evaluation results", "Q1 roadmap meeting", "Bob's request for data"


# EMAIL CONTEXT:
- Subject: {subject}
- From: {sender}
- Date: {sent_datetime}

# Email Body:
{content}

Output:
"""

PROMPT_EMAIL_EPISODIC_MEMORY = """
You are an expert episodic memory generator that creates high-level summaries of email messages.

# TASK:
Generate an episodic memory that captures the overall information of the email below.

Generate the episodic memory in the following format:
EpisodicIndex: [6-8 word summary capturing the main topic of the email]
EpisodicValue: [1-3 sentence summary of the email's content and purpose]

# GUIDELINES:
1. The EpisodicIndex should capture the main topic, event, or purpose of the email. It should be concise (6-8 words) but informative enough to distinguish it from other emails.
2. The EpisodicValue should mention:
   - The main topic or purpose of the email
   - Key participants (sender and recipients by name)
   - The date or time context
   - Overall status, outcome, or action requested
3. Focus on "what is this email about" rather than specific details.
4. Make the summary self-contained and understandable without the original email.
5. Do not include email signatures, greetings, or boilerplate text. Focus on the substantive content of the email.

# Email Context:
- Subject: {subject}
- From: {sender}
- Date: {sent_datetime}

# Email Body:
{content}

Output:
"""


class EmailMemoryBuilder(MemoryBuilder):
    """Memory builder for extracting factual and episodic memories from emails.

    Accepts email body text as a plain string via ``client.add(body_text, builder="email", metadata=...)``.
    Email metadata (subject, sender, recipients, date) is passed via the ``metadata`` dict.
    """

    def __init__(self, cfg: DictConfig, megamem: AgentMemory, model_client: ChatCompletionModel):
        super().__init__(cfg, megamem, model_client)

    def build_memory_entries(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
    ) -> List[MemoryEntry]:
        """Extract factual memories out of an email body.

        Args:
            content: Normalized content from ``normalize_content()``.
                     For emails this is ``{"text": body_text, "segment_messages": None}``.
            metadata: Must include email-specific keys:
                     ``email_subject``, ``email_sender``, ``email_recipients``,
                     ``email_cc``, ``email_sent_datetime``.

        Returns:
            Newly built factual ``MemoryEntry`` objects.
        """
        if isinstance(content, dict):
            content_text = content.get("text", "")
        else:
            content_text = str(content)

        if not content_text.strip():
            logger.warning("Empty email body, skipping memory extraction")
            return []

        md = metadata or {}

        subject = md.get("email_subject", "")
        sender = md.get("email_sender", "")
        sent_datetime = md.get("email_sent_datetime", "")

        # Run the email-specific extraction prompt.
        memories: EmailMemoryOutputs = self._model_client.invoke(
            input=PROMPT_BUILD_EMAIL_MEMORY,
            prompt_args={
                "subject": subject,
                "sender": sender,
                "sent_datetime": sent_datetime,
                "content": content_text,
            },
            response_format=EmailMemoryOutputs,
        )

        memory_entries: List[MemoryEntry] = []
        episodic_id = md.get("episodic_memory_id")
        for pos, mem_out in enumerate(memories.entries):
            try:
                cue_indices_str = ""
                if mem_out.cue_indices:
                    cleaned = self._validate_cue_indices(
                        mem_out.cue_indices, mem_out.index
                    )
                    cue_indices_str = "||".join(cleaned)

                entry = MemoryEntry(
                    memory_type="factual",
                    index=mem_out.index,
                    value=mem_out.value,
                    creation_time=md.get("creation_time", ""),
                    timestamp=md.get("timestamp", ""),
                    cue_indices=cue_indices_str,
                    episodic_memory_ids=[episodic_id] if episodic_id else [],
                )
                memory_entries.append(entry)
            except Exception as exc:
                idx_label = getattr(mem_out, "index", f"email_memory_{pos}")
                logger.error(f"Failed to create memory entry for '{idx_label}': {exc}")
                continue

        logger.debug(
            f"Extracted {len(memory_entries)} factual memories from email "
            f"(subject: {subject!r})"
        )
        return memory_entries

    def generate_episodic_memory(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
    ) -> Optional[MemoryEntry]:
        """Produce a single episodic summary entry for an email.

        Args:
            content: Normalized content from ``normalize_content()``.
            metadata: Must include email-specific context keys.

        Returns:
            The episodic ``MemoryEntry`` or ``None`` when generation failed.
        """
        try:
            content_text = content.get("text", "") if isinstance(content, dict) else str(content)

            if not content_text.strip():
                logger.warning("Empty email body, skipping episodic memory generation")
                return None

            md = metadata or {}
            subject = md.get("email_subject", "")
            sender = md.get("email_sender", "")
            sent_datetime = md.get("email_sent_datetime", "")

            episodic_output: EmailEpisodicMemoryOutput = self._model_client.invoke(
                input=PROMPT_EMAIL_EPISODIC_MEMORY,
                prompt_args={
                    "subject": subject,
                    "sender": sender,
                    "sent_datetime": sent_datetime,
                    "content": content_text,
                },
                response_format=EmailEpisodicMemoryOutput,
            )

            return MemoryEntry(
                memory_type="episodic",
                index=f"[EPISODIC] {episodic_output.episodic_index}",
                value=episodic_output.episodic_value,
                creation_time=md.get("creation_time", ""),
                timestamp=md.get("timestamp", ""),
            )

        except Exception as exc:
            logger.warning(f"Failed to generate episodic memory for email: {exc}")
            return None

    def should_process_thread(self, emails: List[NormalizedEmail]) -> bool:
        """Ask the LLM whether a thread carries enough substance for extraction.

        Args:
            emails: Thread messages in chronological order.

        Returns:
            ``True`` if the thread should be processed, ``False`` to skip it.
        """
        if not emails:
            return False

        first_email = emails[0]
        last_email = emails[-1]
        subject = first_email.subject or "(no subject)"

        # Collect every participant that appears anywhere in the thread.
        participants: set = set()
        for em in emails:
            participants.add(f"{em.sender_name} <{em.sender_address}>")
            for recipient in em.to_recipients + em.cc_recipients:
                name = recipient.get("name", recipient.get("address", ""))
                if name:
                    participants.add(name)

        if len(emails) > 1:
            date_range = f"{first_email.sent_datetime} -> {last_email.sent_datetime}"
        else:
            date_range = first_email.sent_datetime

        # Build a compact thread preview (truncate each body so the prompt stays small).
        parts = [
            f"[{pos + 1}] From: {em.sender_name} | Date: {em.sent_datetime}\n{em.body_text.strip()[:300]}"
            for pos, em in enumerate(emails)
        ]
        thread_content = "\n\n".join(parts)

        try:
            result: EmailFilterOutput = self._model_client.invoke(
                input=PROMPT_EMAIL_THREAD_FILTER,
                prompt_args={
                    "subject": subject,
                    "participants": ", ".join(sorted(participants)),
                    "date_range": date_range,
                    "num_emails": str(len(emails)),
                    "thread_content": thread_content,
                },
                response_format=EmailFilterOutput,
            )

            if not result.has_extractable_content:
                logger.info(
                    f"Skipping thread (subject: {subject!r}, "
                    f"{len(emails)} emails): no extractable content"
                )
            return result.has_extractable_content

        except Exception as exc:
            logger.warning(f"Thread filter LLM call failed, proceeding with extraction: {exc}")
            return True

    def _email_prompt_args(self, email: NormalizedEmail, content_text: str) -> Dict:
        """Build LLM prompt arguments directly from a NormalizedEmail."""
        return {
            "subject": email.subject,
            "sender": f"{email.sender_name} <{email.sender_address}>",
            "recipients": ", ".join(
                r.get("name", r.get("address", "")) for r in email.to_recipients
            ),
            "sent_datetime": email.sent_datetime,
            "content": content_text,
        }

    def build_from_emails(
        self,
        emails: List[NormalizedEmail],
    ) -> List[MemoryEntry]:
        """Build memories from an email thread (or a lone email)."""
        if not emails or not self.should_process_thread(emails):
            return []

        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)

        # Process every email in the surviving thread.
        all_entries: List[MemoryEntry] = []
        for email in emails:
            entries, _surviving = self._build_single_email(email, enable_episodic)
            all_entries.extend(entries)

        return all_entries

    def _build_single_email(
        self,
        email: NormalizedEmail,
        enable_episodic: bool = False,
    ) -> tuple:
        """Extract memories from one email (no thread-level filter).

        Args:
            email: Email payload to mine for memories.
            enable_episodic: When ``True``, also build an episodic summary.

        Returns:
            ``(memory_entries, surviving_indices)`` where the first item is
            the freshly extracted entries and the second is the index that
            actually landed in the store after upsert.
        """
        subject = email.subject.strip()
        body = email.body_text.strip()

        if not body:
            logger.warning("Empty email body, skipping memory extraction")
            return []

        content_text = f"Subject: {subject}\n\nContent: {body}" if subject else f"Content: {body}"
        prompt_args = self._email_prompt_args(email, content_text)
        creation_time = get_current_timestamp()
        timestamp = email.sent_datetime

        # ------ Step 1: optional episodic summary ------------------------
        episodic_memory_id = None
        if enable_episodic:
            try:
                episodic_output: EmailEpisodicMemoryOutput = self._model_client.invoke(
                    input=PROMPT_EMAIL_EPISODIC_MEMORY,
                    prompt_args=prompt_args,
                    response_format=EmailEpisodicMemoryOutput,
                )
                episodic_entry = MemoryEntry(
                    memory_type="episodic",
                    index=f"[EPISODIC] {episodic_output.episodic_index}",
                    value=episodic_output.episodic_value,
                    creation_time=creation_time,
                    timestamp=timestamp,
                )
                self.megamem.add(episodic_entry)
                episodic_memory_id = episodic_entry.index
            except Exception as exc:
                logger.warning(f"Failed to generate episodic memory for email: {exc}")

        # ------ Step 2: factual memory entries ---------------------------
        memories: EmailMemoryOutputs = self._model_client.invoke(
            input=PROMPT_BUILD_EMAIL_MEMORY,
            prompt_args=prompt_args,
            response_format=EmailMemoryOutputs,
        )

        memory_entries: List[MemoryEntry] = []
        for pos, mem_out in enumerate(memories.entries):
            try:
                cue_indices_str = ""
                if mem_out.cue_indices:
                    cleaned = self._validate_cue_indices(
                        mem_out.cue_indices, mem_out.index
                    )
                    cue_indices_str = "||".join(cleaned)

                entry = MemoryEntry(
                    memory_type="factual",
                    index=mem_out.index,
                    value=mem_out.value,
                    creation_time=creation_time,
                    timestamp=timestamp,
                    cue_indices=cue_indices_str,
                    episodic_memory_ids=[episodic_memory_id] if episodic_memory_id else [],
                )
                memory_entries.append(entry)
            except Exception as exc:
                idx_label = getattr(mem_out, "index", f"email_memory_{pos}")
                logger.error(f"Failed to create memory entry for '{idx_label}': {exc}")

        # ------ Step 3: upsert each new entry ----------------------------
        surviving_indices: List[str] = [
            self.upsert_memory_entry(entry=entry) for entry in memory_entries
        ]

        return memory_entries, surviving_indices

    @staticmethod
    def _validate_cue_indices(cue_indices: List[str], primary_index: str) -> List[str]:
        """Sanitize LLM-generated cue indices."""
        validated: List[str] = []
        seen: set = set()
        primary_lower = primary_index.lower()

        for raw in cue_indices:
            cue = raw.strip()
            if not cue:
                continue
            cue_lower = cue.lower()
            if cue_lower in seen or cue_lower == primary_lower or len(cue.split()) < 2:
                continue
            seen.add(cue_lower)
            validated.append(cue)

        return validated[:3]
