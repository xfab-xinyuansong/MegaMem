"""
Retrieval Planner

Inspects user queries and emits step-based execution plans built from three
primitives:
  - FILTER:           metadata-only filtering through a ChromaDB where clause
  - RESOLVE:          structured extraction over FILTER results (no embeddings)
  - SEMANTIC_SEARCH:  embedding-based similarity search

The planner emits flat, typed filter fields — NOT raw ChromaDB where clause
dicts. The executor (in AgentMemory) translates them into ChromaDB where
clauses.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.memory_entry import MemoryEntry
from megamem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class RetrievalStep(BaseModel):
    """One step inside the retrieval plan."""
    step_id: str = Field(description="Step identifier: 'S1', 'S2', etc.")
    op: str = Field(description="Operation: 'FILTER', 'SEMANTIC_SEARCH', or 'RESOLVE'")
    scope: Optional[str] = Field(
        default="all_sources",
        description="'all_sources' (default) or 'filtered_results' (operates on output of previous step)"
    )

    # --- Shared metadata fields (available on FILTER and SEMANTIC_SEARCH) ---
    data_type: Optional[str] = Field(
        default=None,
        description="Source type: 'mail', 'doc', or null. Available on FILTER and SEMANTIC_SEARCH."
    )
    timestamp_after: Optional[str] = Field(
        default=None,
        description="ISO date 'YYYY-MM-DD' lower bound (inclusive). Available on FILTER and SEMANTIC_SEARCH."
    )
    timestamp_before: Optional[str] = Field(
        default=None,
        description="ISO date 'YYYY-MM-DD' upper bound (exclusive). Available on FILTER and SEMANTIC_SEARCH."
    )

    # --- String-filtered fields (post-filtered in Python on FILTER and SS(source_cues)) ---
    sender: Optional[str] = Field(
        default=None,
        description="Sender/author name (lowercase, first name or full name). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring. Only set when query explicitly names a sender/author."
    )
    recipients: Optional[str] = Field(
        default=None,
        description="Recipient name (lowercase, first name or full name). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring. Only set when query explicitly asks about emails sent TO someone."
    )
    author: Optional[str] = Field(
        default=None,
        description="Document author name (lowercase). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring. Use for doc sources when data_type='doc'."
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title or mail subject (lowercase). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring."
    )
    participants: Optional[str] = Field(
        default=None,
        description="Chat participant name (lowercase). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring. Use for chat sources when data_type='chat'."
    )
    topic: Optional[str] = Field(
        default=None,
        description="Chat channel or thread topic (lowercase). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via substring. Use for chat sources when data_type='chat'."
    )
    conversation_type: Optional[str] = Field(
        default=None,
        description="Chat sub-type: 'meeting', 'chat', or 'channel' (lowercase). "
                    "Available on FILTER and SEMANTIC_SEARCH(target='source_cues'). "
                    "Matched via exact value. Use for chat sources when data_type='chat'."
    )

    # --- SEMANTIC_SEARCH fields ---
    query_text: Optional[str] = Field(
        default=None,
        description="For SEMANTIC_SEARCH: semantic query string, stripped of source/temporal qualifiers"
    )
    target: Optional[str] = Field(
        default=None,
        description="For SEMANTIC_SEARCH: 'source_cues' (search source descriptions to identify relevant sources) "
                    "or 'primary_memories' (search extracted facts/details for specific information)"
    )

    # --- RESOLVE fields ---
    return_mode: Optional[str] = Field(
        default=None,
        description="For RESOLVE: 'metadata_summary' (return source cue descriptions + metadata fields) "
                    "or 'full_content' (return all linked primary memories from matched sources)"
    )
    metadata_fields: Optional[List[str]] = Field(
        default=None,
        description="For RESOLVE: which metadata fields to extract, e.g. ['subject', 'sender', 'date']. "
                    "If null, return all available fields. Only used with return_mode='metadata_summary'."
    )


class RetrievalPlan(BaseModel):
    """Plan emitted by the retrieval-planning LLM."""
    normalized_query: str = Field(
        description="Cleaned/restated version of the user query"
    )
    steps: List[RetrievalStep] = Field(
        description="Ordered list of 1-3 steps to execute"
    )
    assumptions: Optional[List[str]] = Field(
        default_factory=list,
        description="Assumptions made by the planner"
    )
    reasoning: str = Field(
        description="Brief explanation of why this plan was chosen"
    )




"""
Based on the below rules:
possible plans include:
- FILTER → RESOLVE(metadata_summary)
- FILTER → RESOLVE(full_content)
- SEMANTIC_SEARCH(source_cues) → RESOLVE(metadata_summary)
- SEMANTIC_SEARCH(source_cues) → RESOLVE(full_content)
- SEMANTIC_SEARCH(source_cues) → SEMANTIC_SEARCH(primary_memories)
- SEMANTIC_SEARCH(primary_memories)
"""

PROMPT_RETRIEVAL_PLANNER = """

You are a RETRIEVAL planner for a memory system that stores information extracted from different sources, not an answering agent. Your sole job is to retrieve the most relevant memories for the downstream agent. 

Therefore:
Never decompose a multi-part user request into separate retrieval steps per sub-question
Complex queries spanning multiple source types should use 1-2 broad retrieval steps, not one step per source type
The maximum is 2 steps. If you cannot express the retrieval in 2 steps, choose the single broadest retrieval that captures the most relevant information (typically SS(primary_memories) with a well-crafted query_text)

## MEMORY ARCHITECTURE

The system has two layers of memory:
- **Source cues**: High-level descriptions of each ingested source. They carry source-level metadata (sender, subject, date, etc.) and are linked to their primary memories.
- **Primary memories**: Individual facts and details extracted from sources. Each is linked back to its source cue. They do NOT carry source-level metadata.

Use source cues to identify WHICH sources are relevant.
Use primary memories to find specific CONTENT within sources.

Important: Only source cues carry source-level metadata (sender, recipients, author, etc.). Primary memories do NOT carry this metadata. Therefore, person-based filtering is only possible at the source cue level, not at the primary memory level.

## AVAILABLE METADATA PER SOURCE TYPE

Source cues store metadata that varies by source type:
- mail: data_type, sender, recipients, subject, date 
- doc: data_type, author, title, date 
- teams: data_type, participants, topic, conversation_type, date

Metadata Fields:
- `data_type`: "mail", "doc", or "teams" — narrows search to a specific source type
- `sender` (mail) / `author` (doc): the person who created or sent the source.
- `recipients`: the TO field (mail only)
- `subject` (mail) / `title` (doc): the heading of the source
- `participants` (teams): people in the chat/channel thread
- `topic` (teams): the Teams channel or thread topic name
- `conversation_type` (teams): sub-type of chat — "meeting", "chat", or "channel"
- `date`: ISO date string

The following metadata fields can be used on FILTER and SEMANTIC_SEARCH(target="source_cues") steps:
- data_type (matched via chromadb where)
- timestamp_after (matched via chromadb where)
- timestamp_before (matched via chromadb where)
- sender (matched via substring — e.g. "sarah" matches "Sarah Johnson <sarah@company.com>")
- recipients (matched via substring)
- author (matched via substring)
- title  (matched via substring)
- participants (matched via substring)
- topic (matched via substring)
- conversation_type (matched via exact value — "meeting", "chat", or "channel")


# DATA TYPE DETECTION
- "mail" signals: email, mail, inbox
- "doc" signals: document, file, report, notes, PDF, spreadsheet, slides, paper
- "teams" signals: Teams, chat, channel, message, thread, meeting chat
- Words like "said", "say", "tell", "told", "mentioned", "discussed", "talked about" do NOT imply email — they describe verbal/written communication in general. Do not infer data_type from these words.
- If no clear data type signal, set data_type to null (do not guess), its associated fields are not set as well.

** First identify the data type, then include only the relevant metadata fields which be inferred through the query for that data type.

## TASK
Given a user query, produce a retrieval plan consisting of 1-3 steps. Each step uses one of three primitives.

## PRIMITIVES

1. **FILTER** — Metadata-only lookup (no embedding search).
   Use ONLY for pure listing/enumeration queries where no content/topic matching is needed.
   Fields (decided based on data_type):
   - `data_type`, `timestamp_after`, `timestamp_before` — passed to the database as exact/range filters
   - `sender`, `recipients` — applied as substring post-filters

2. **SEMANTIC_SEARCH** — Embedding-based similarity search. Finds content by meaning.
   Fields:
   - `query_text` (required): the content/topic portion of the query (strip source-type words, dates, person names, filler)
   - `target` (required):
     - `"source_cues"` — search source descriptions to identify relevant sources. Supports 'data_type' along with its associated metadata.
     - `"primary_memories"` — search extracted facts/details
   - `scope` (required):
     - `"all_sources"` — search the entire collection (default)
     - `"filtered_results"` — search within the output of the previous step
   -  IMPORTANT: Metadata fields as defined in "Metadata Fields" section above are only used when target="source_cues", never with target="primary_memories" because primary memories do not carry source-level metadata.
     

3. **RESOLVE** — Structured extraction from matched source cues. No embedding search.
   Must follow a FILTER step or a SEMANTIC_SEARCH(target="source_cues") step.
   Never appears as the first step.
   Fields available to attach
   - `return_mode`:
     - `"metadata_summary"`: return source cue descriptions + selected metadata. Use for listing/enumeration.
     - `"full_content"`: return ALL linked primary memories from matched sources. Use for summarization.

# PLANNING RULES
(for simplicity referring to all possible metadata fields as metadata)

1. Do not assume the data type and other related metadata unless the query explictly mentions it. 

2. While creating the metadata fields refer to "## AVAILABLE METADATA PER SOURCE TYPE" and "## DATA TYPE DETECTION".
   The metadata fields attach only to FILTER or SEMANTIC_SEARCH(target="source_cues") steps and never to RESOLVE or SS(primary_memories) because primary memories do not carry metadata. 

3. Only include metadata fields that are explicitly mentioned in the query. 
   For example, if the query is "What did Sarah say about the budget?", since we cannot infer the Source type from the query,
   'Sarah' could be the author of a document as well as the sender of a message or email. So it is better to keep the metadata empty.

4. If the data type can be inferred, include the relevant metadata fields for that data type as per "## AVAILABLE METADATA PER SOURCE TYPE". 
   For example, if the query is "What did Sarah say about the budget in her email?", we can infer that the data type is 'mail' and hence we can include the 'sender' field with value 'Sarah' as metadata in the SEMANTIC_SEARCH step.

5. When ALL constraints can be expressed purely as metadata fields (data_type, sender, recipients, timestamps, etc.) and the query has NO content/topic to search for:
   a. Use FILTER to narrow down source cues by metadata
   b. Follow with RESOLVE:
      - return_mode="metadata_summary" for listing/enumeration (e.g., "list emails from Sarah")
      - return_mode="full_content" for summarization (e.g., "summarize the email sent by Carol")
   Key test: if you cannot extract a meaningful query_text (a topic/subject beyond source-type words, person names, and dates), use FILTER — not SEMANTIC_SEARCH.
   FILTER is exclusively for pure enumeration (no content matching), always paired with RESOLVE. If you have metadata constraints AND a content/topic to search for, 
   use SEMANTIC_SEARCH(target="source_cues") with both the metadata fields and query_text — never FILTER followed by SEMANTIC_SEARCH(primary_memories). 
   ""FILTER → SEMANTIC_SEARCH is not a valid plan.""

6. For queries that require listing/enumeration but also include a content/topic reference (e.g., "list the documents mentioning about AI agents usage")
   a. Use SEMANTIC_SEARCH(target="source_cues",metadata,query_text) to find relevant sources based on content
   b. Follow with RESOLVE(return_mode)

7. For other queries first determine if the user is asking a broad or a very specific question.
   a For broad queries that include a content/topic component beyond just metadata (e.g., "Summarize Sarah's email about the budget" has topic "budget"; "Which document mentions cloud migration?" has topic "cloud migration"):
   - Use SEMANTIC_SEARCH(target="source_cues",metadata,query_text) to find relevant sources by content similarity
   - then RESOLVE(return_mode=metadata_summary/full_content)
   Note: if the query is broad but has NO content/topic (e.g., "Summarize Carol's email"), rule 5 applies — use FILTER, not SEMANTIC_SEARCH.
   b.For specific queries like: "What was the Q3 budget?" or "What did Sarah say about the deadline?"
   - Use SEMANTIC_SEARCH(target="primary_memories") to find specific facts/details.
   - If person names cannot be set as metadata fields (because data_type is unknown), keep them in query_text.
   - A plan may contain at most one SEMANTIC_SEARCH(target='primary_memories') step. SEMANTIC_SEARCH(primary_memories) --> SEMANTIC_SEARCH(primary_memories) is never valid because the second step cannot meaningfully re-scope against the first.

8. Some queries might require two SEMANTIC_SEARCH steps: first to find relevant sources via their cues, then to drill into primary memories for specific content. 
   Prefer this pattern when the query asks about a specific topic that might only appear in the detailed content of a source, not in its high-level description.
   - Use SEMANTIC_SEARCH(target="source_cues",metadata,query_text) to find relevant source cues
   - Then use SEMANTIC_SEARCH(target="primary_memories", scope="filtered_results") to find specific information.

9. RESOLVE must follow FILTER or SEMANTIC_SEARCH(target="source_cues"). Never use RESOLVE as the first step.

10. SEMANTIC_SEARCH(target="source_cues") is NEVER the final step. It MUST always be followed by RESOLVE or SEMANTIC_SEARCH(target="primary_memories"). If you find yourself wanting a single-step plan 
    example: the query is very specific with no reference to any kind of source, use SEMANTIC_SEARCH(target="primary_memories") directly rather than going through source cues. But if you do use SEMANTIC_SEARCH(target="source_cues"), follow it with another step to extract the information.

11. FILTER is never terminal. Every FILTER step MUST be followed by RESOLVE. A plan that ends with FILTER is invalid.

12. A plan may contain at most ONE FILTER step. If the query asks about multiple source types (e.g., "emails and Teams chats"), omit data_type to search across all source types rather than creating separate FILTER steps per type. FILTER → FILTER is never valid.

# TEMPORAL NORMALIZATION
Today's date: {today}
- "last week" → timestamp_after: Monday of previous week, timestamp_before: Monday of current week
- "yesterday" → timestamp_after: yesterday, timestamp_before: today
- "last month" → timestamp_after: 1st of previous month, timestamp_before: 1st of current month
- "this week" → timestamp_after: Monday of current week, timestamp_before: null
- "after January 2026" → timestamp_after: "2026-01-31"
- "before February 2026" → timestamp_before: "2026-02-01"
- If no temporal reference → leave both null

# SEMANTIC QUERY EXTRACTION
When writing `query_text` for SEMANTIC_SEARCH, extract ONLY the content portion:
- Remove source-type words (email, document, file, mail, chat, channel, Teams)
- Remove temporal qualifiers (last week, yesterday, after Feb 10)
- Remove person names ONLY when they are set as metadata fields (sender/recipients). Keep person names in query_text when data_type is unknown and person fields cannot be used.
- Remove filler (show me, find, list, please)
- Keep entity names, topics, and action words

------------------------------------------------------------------------------------------------
# EXAMPLES

1. Query: "List emails from Sarah from previous week"
Plan:
- S1: FILTER(data_type="mail", timestamp_after="2026-02-09", timestamp_before="2026-02-16", sender="sarah")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "Pure enumeration. No content matching needed. FILTER narrows by data_type=mail. RESOLVE returns source descriptions."

2. Query: "Summarize the project status document shared last month"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="doc", timestamp_after="2026-01-01", timestamp_before="2026-02-01", query_text="project status")
- S2: RESOLVE(scope="filtered_results", return_mode="full_content")
Reasoning: "User wants full summarization of a document. No person filter needed. SS(source_cues) finds the document by content, RESOLVE extracts full content."

3. Query: "What emails did I receive this week?"
Plan:
- S1: FILTER(data_type="mail", timestamp_after="2026-02-16")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "Pure enumeration of this week's emails. No content matching needed. FILTER by type and date, RESOLVE lists summaries."

4. Query: "Summarize the email sent by Carol"
Plan:
- S1: FILTER(data_type="mail", sender="carol")
- S2: RESOLVE(scope="filtered_results", return_mode="full_content")
Reasoning: "Summarization request but all constraints are pure metadata (data_type=mail, sender=carol). No content/topic to search for — 'email sent by Carol' has no meaningful query_text. FILTER narrows by metadata, RESOLVE extracts full content for summarization."

5. Query: "Summarize Sarah's email about the budget"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="mail", sender="sarah", query_text="budget")
- S2: RESOLVE(scope="filtered_results", return_mode="full_content")
Reasoning: "Content query with a person constraint. SS(source_cues) finds budget-related emails by semantic similarity and post-filters to Sarah's emails. RESOLVE extracts full content for summarization."

6. Query: "Which document mentions cloud migration?"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="doc", query_text="cloud migration")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "User wants to identify which document mentions a topic. SS(source_cues) finds documents about cloud migration. RESOLVE returns metadata summaries showing which documents matched."

7. Query: "What does the project report say about resource allocation?"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="doc", query_text="project report")
- S2: SEMANTIC_SEARCH(target="primary_memories", scope="filtered_results", query_text="resource allocation")
Reasoning: "User asks about specific content within a source. SS(source_cues) identifies the project report. SS(primary_memories) drills into its extracted facts to find resource allocation details."

8. Query: "What did Sarah say about the deadline?"
Plan:
- S1: SEMANTIC_SEARCH(target="primary_memories", scope="all_sources", query_text="Sarah deadline")
Reasoning: "Specific fact query. No data_type can be inferred — 'say' does not imply email. Since data_type is unknown, person fields (sender/recipients) cannot be set either. Search primary memories directly for deadline-related facts. Person name 'Sarah' is kept in query_text to leverage semantic similarity."

9. Query: "What was the Q3 budget?"
Plan:
- S1: SEMANTIC_SEARCH(target="primary_memories", scope="all_sources", query_text="Q3 budget amount")
Reasoning: "Pure content question with no source, date, or person constraints. User asks for a specific fact, so search primary memories directly."

10. Query: "Show me emails sent to David last month"
Plan:
- S1: FILTER(data_type="mail", timestamp_after="2026-01-01", timestamp_before="2026-02-01", recipients="david")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "Pure enumeration with recipient filter. FILTER by type, date, and recipient. RESOLVE lists summaries."

11. Query: "What was the Atlas timeline email sent by Sarah last week?"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="mail", sender="sarah", timestamp_after="2026-02-09", timestamp_before="2026-02-16", query_text="Atlas timeline")
- S2: SEMANTIC_SEARCH(target="primary_memories", scope="filtered_results", query_text="Atlas timeline")
Reasoning: "Person + specific fact query. SS(source_cues) finds timeline-related emails by semantic similarity, post-filters to Sarah's emails within the date range. SS(primary_memories) drills into those sources for the exact timeline details."

Query: "Tell me about Q4 planning discussed in emails last week"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="mail", timestamp_after="2026-02-16", timestamp_before="2026-02-23", query_text="Q4 planning")
- S2: SEMANTIC_SEARCH(target="primary_memories", scope="filtered_results", query_text="Q4 planning")
Reasoning: "SS(source_cues) finds Q4-planning emails by content within the date range. SS(primary_memories) drills into primary memories for detailed information."

Query: "List all documents added this month"
Plan:
- S1: FILTER(data_type="doc", timestamp_after="2026-02-01")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "Pure enumeration of documents. No content matching needed. FILTER by type and date, RESOLVE lists document summaries."

14. Query: "What was discussed in the Engineering-Core channel about deployment?"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="teams", conversation_type="channel", topic="engineering-core", query_text="deployment")
- S2: SEMANTIC_SEARCH(target="primary_memories", scope="filtered_results", query_text="deployment")
Reasoning: "User asks about specific content in a chat channel. SS(source_cues) finds channel threads in Engineering-Core about deployment. SS(primary_memories) drills into those threads for deployment details."

15. Query: "List all Teams chats from last week"
Plan:
- S1: FILTER(data_type="teams", timestamp_after="2026-02-23", timestamp_before="2026-03-02")
- S2: RESOLVE(scope="filtered_results", return_mode="metadata_summary")
Reasoning: "Pure enumeration of chat threads. No content matching needed. FILTER by data_type=chat and date range, RESOLVE lists thread summaries."

16. Query: "What did the team discuss about the release timeline in Teams?"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="teams", query_text="release timeline")
- S2: RESOLVE(scope="filtered_results", return_mode="full_content")
Reasoning: "User wants to summarize chat content about a topic. SS(source_cues) finds chat threads about release timeline. RESOLVE extracts full content for summarization."

17. Query: "Summarize what was said in my meeting chats with the design team last week"
Plan:
- S1: SEMANTIC_SEARCH(target="source_cues", data_type="teams", conversation_type="meeting", participants="design team", query_text="meeting discussion", timestamp_after="2026-02-23", timestamp_before="2026-03-02")
- S2: RESOLVE(scope="filtered_results", return_mode="full_content")
Reasoning: "User wants meeting chat summaries. SS(source_cues) finds meeting-type chat threads with the design team in the date range. RESOLVE returns full content for summarization."

# USER QUERY
{query}

Produce the retrieval plan.
"""


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------

class RetrievalPlanner:
    """
    Inspects user queries and emits a step-based execution plan.

    Usage:
        planner = RetrievalPlanner(cfg)
        plan = planner.plan("Summarize the email from Sarah last week")
        # plan.steps → [FILTER(data_type=mail, ...), SEMANTIC_SEARCH(...)]
    """

    def __init__(self, cfg: DictConfig, model_client: Optional[ChatCompletionModel] = None):
        self.cfg = cfg
        self._model_client = model_client or ChatCompletionModel(cfg)

    def plan(self, query: str) -> RetrievalPlan:
        """
        Build a retrieval plan from a natural-language user query.

        Args:
            query: Natural language user query

        Returns:
            RetrievalPlan with ordered steps to execute
        """
        today = datetime.now().strftime("%Y-%m-%d")

        prompt_args = {
            "query": query,
            "today": today,
        }

        try:
            result: RetrievalPlan = self._model_client.invoke(
                input=PROMPT_RETRIEVAL_PLANNER,
                prompt_args=prompt_args,
                response_format=RetrievalPlan,
            )
            logger.info(
                f"Retrieval plan for '{query[:50]}...': "
                f"{len(result.steps)} steps — {result.reasoning[:80]}"
            )
            return result

        except Exception as e:
            logger.warning(f"Retrieval planner failed: {e}. Falling back to SEMANTIC_SEARCH only.")
            return RetrievalPlan(
                normalized_query=query,
                steps=[
                    RetrievalStep(
                        step_id="S1",
                        op="SEMANTIC_SEARCH",
                        scope="all_sources",
                        query_text=query,
                    )
                ],
                assumptions=["Planner LLM call failed; using full semantic search as fallback"],
                reasoning="Fallback: planner failed, defaulting to unfiltered semantic search.",
            )


# ---------------------------------------------------------------------------
# Where clause builder (used by the executor in AgentMemory)
# ---------------------------------------------------------------------------

def build_where_clause(step: RetrievalStep) -> Optional[Dict]:
    """
    Translate a RetrievalStep's flat filter fields into a ChromaDB where clause.

    Only handles ChromaDB-native operators:
      - data_type exact match ($eq)
      - timestamp_after → timestamp_unix $gte
      - timestamp_before → timestamp_unix $lt

    String-based filters (sender, recipients) flow through
    get_string_filters() + apply_string_filters() instead, since
    ChromaDB's $contains operator does not perform substring matching on
    metadata fields (it is wired up for list membership / where_document).

    Args:
        step: RetrievalStep with flat filter fields

    Returns:
        ChromaDB-compatible where dict, or None if no filters apply.
    """
    conditions: List[Dict] = []

    if step.data_type:
        conditions.append({"data_type": {"$eq": step.data_type}})

    if step.timestamp_after:
        unix_ts = _date_to_unix(step.timestamp_after)
        if unix_ts:
            conditions.append({"timestamp_unix": {"$gte": unix_ts}})

    if step.timestamp_before:
        unix_ts = _date_to_unix(step.timestamp_before)
        if unix_ts:
            conditions.append({"timestamp_unix": {"$lt": unix_ts}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# Maps the RetrievalStep attribute name to the output key used in the string filters dict.
_STRING_FILTER_FIELDS = (
    "sender", "recipients", "author", "title",
    "participants", "topic", "conversation_type",
)


def get_string_filters(step: RetrievalStep) -> Dict[str, Any]:
    """
    Extract the string-based post-filters from a RetrievalStep.

    These get applied in Python after ChromaDB returns its raw results,
    because ChromaDB's $contains does not do substring matching on metadata.
    All values are lowercased for case-insensitive matching.

    Returns:
        Dict with any of: sender, recipients, author, title. Empty dict when no string filters apply.
    """
    filters: Dict[str, Any] = {}

    for attr in _STRING_FILTER_FIELDS:
        raw = getattr(step, attr)
        if raw:
            filters[attr] = raw.lower()

    return filters


def apply_string_filters(
    entries: List,
    string_filters: Dict[str, Any],
) -> List:
    """
    Apply Python-side substring filters to a list of MemoryEntry objects.

    Walks each entry's extra_metadata for substring containment:
      - sender: entry.extra_metadata["sender"] contains the filter value
      - recipients: entry.extra_metadata["recipients"] contains the filter value
      - author: entry.extra_metadata["author"] contains the filter value
      - title: entry.extra_metadata["title"] or ["subject"] contains the filter value

    Args:
        entries: List of MemoryEntry objects to filter
        string_filters: Dict from get_string_filters()

    Returns:
        Filtered list of MemoryEntry objects
    """
    if not string_filters:
        return entries

    result = []
    for entry in entries:
        extra = entry.extra_metadata or {}

        # sender substring
        if "sender" in string_filters:
            sender_val = (extra.get("sender", "") or "").lower()
            if string_filters["sender"].lower() not in sender_val:
                continue

        # recipients substring
        if "recipients" in string_filters:
            recip_val = (extra.get("recipients", "") or "").lower()
            if string_filters["recipients"].lower() not in recip_val:
                continue

        # author substring
        if "author" in string_filters:
            author_val = (extra.get("author", "") or "").lower()
            if string_filters["author"].lower() not in author_val:
                continue

        # title substring (also looks at "subject")
        if "title" in string_filters:
            title_val = (extra.get("title", "") or extra.get("subject", "") or "").lower()
            if string_filters["title"].lower() not in title_val:
                continue

        # participants substring
        if "participants" in string_filters:
            participants_val = (extra.get("participants", "") or "").lower()
            if string_filters["participants"].lower() not in participants_val:
                continue

        # topic substring
        if "topic" in string_filters:
            topic_val = (extra.get("topic", "") or "").lower()
            if string_filters["topic"].lower() not in topic_val:
                continue

        # conversation_type exact match
        if "conversation_type" in string_filters:
            ct_val = (extra.get("conversation_type", "") or "").lower()
            if ct_val != string_filters["conversation_type"].lower():
                continue

        result.append(entry)

    return result


def _date_to_unix(date_str: str) -> int:
    """Convert a YYYY-MM-DD date string into a Unix timestamp; returns 0 on failure."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Planner executor helpers (used by AgentMemory._execute_planner_query)
# ---------------------------------------------------------------------------

def resolve_source_cues(
    source_cues: List[MemoryEntry],
    return_mode: str,
    metadata_fields: Optional[List[str]] = None,
) -> List[MemoryEntry]:
    """
    Return deduplicated source-cue entries for a RESOLVE step.

    Both return modes wrap a source cue in a MemoryEntry; the memory_type
    tag conveys the caller's intent to the application layer:

    - return_mode="metadata_summary" --> memory_type="resolve_list"
      For listing/enumeration queries ("list emails from Sarah").


    - return_mode="full_content" --> memory_type="resolve_content"
      For summarization queries ("summarize doc A").

    Args:
        source_cues: Source cue entries from FILTER or SS(source_cues).
        return_mode: "metadata_summary" or "full_content".
        metadata_fields: Optional list of metadata field names to keep
            (only used when return_mode="metadata_summary").

    Returns:
        Deduplicated list of MemoryEntry objects.
    """
    memory_type = (
        "resolve_list" if return_mode == "metadata_summary"
        else "resolve_content"
    )

    results: List[MemoryEntry] = []
    seen: set = set()

    for cue in source_cues:
        if not cue.is_cue_index():
            continue
        if cue.index in seen:
            continue
        seen.add(cue.index)

        extra = cue.extra_metadata or {}
        if return_mode == "metadata_summary" and metadata_fields:
            extra = {k: v for k, v in extra.items() if k in metadata_fields}

        results.append(MemoryEntry(
            index=cue.index,
            value=cue.index,
            data_type=cue.data_type or "",
            timestamp_unix=cue.timestamp_unix or 0,
            memory_type=memory_type,
            extra_metadata=extra,
        ))

    logger.info(
        f"RESOLVE({return_mode}): returned {len(results)} source entries "
        f"from {len(source_cues)} source cues"
    )

    return results


def merge_where_clauses(clause_a: Optional[Dict], clause_b: Optional[Dict]) -> Optional[Dict]:
    """
    Combine two ChromaDB where clauses into a single $and clause.
    Flattens nested $and conditions so we don't end up with deeply nested dicts.

    SS steps use this to pair their own data_type/timestamps filter with
    FILTER's accumulated where clause.

    Returns None when both inputs are None.
    """
    if not clause_a:
        return clause_b
    if not clause_b:
        return clause_a

    conditions: List[Dict] = []
    for clause in (clause_a, clause_b):
        if "$and" in clause:
            conditions.extend(clause["$and"])
        else:
            conditions.append(clause)
    return {"$and": conditions}


def build_source_cue_filter(accumulated_where: Optional[Dict]) -> Dict:
    """
    Combine a FILTER's accumulated where clause with the source-cue restriction.

    Ensures we only match source cue entries (cue_type == "source"),
    while still honoring any data_type / timestamp / sender filters from
    the plan.

    Args:
        accumulated_where: Where clause built by FILTER step, or None

    Returns:
        ChromaDB-compatible where dict scoped to source cues
    """
    source_cue_where = {"cue_type": {"$eq": "source"}}
    if not accumulated_where:
        return source_cue_where
    if "$and" in accumulated_where:
        return {"$and": accumulated_where["$and"] + [source_cue_where]}
    return {"$and": [accumulated_where, source_cue_where]}
