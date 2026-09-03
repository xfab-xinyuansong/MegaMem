"""Local document ingestion and memory operations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from omegaconf import DictConfig

from megamem.builder.document_memory_builder import DocumentMemoryBuilder
from megamem.builder.chat_memory_builder import ChatMemoryBuilder, NormalizedChatMessage
from megamem.builder.email_memory_builder import EmailMemoryBuilder, NormalizedEmail
from megamem.builder.memory_builder import MemoryBuilder
from megamem.builder.memory_builder_registry import MemoryBuilderRegistry
from megamem.core.memory import AgentMemory, QueryMode
from megamem.core.memory_entry import MemoryEntry
from megamem.core.segment import Segment
from megamem.core.source_cue_generator import SourceCueGenerator
from megamem.processors.base_processor import detect_file_type, is_supported_file_type
from megamem.processors.teams_chat_processor import TeamsChatProcessor
from megamem.processors.processor_registry import ProcessorRegistry
from megamem.utils.llm import ChatCompletionModel
from megamem.utils.log import log_segments
from megamem.utils.misc import merge_metadata

logger = logging.getLogger(__name__)


class LocalMemoryClient:
    """High-level operations over an in-process, user-scoped memory store."""

    def __init__(
        self,
        cfg: DictConfig,
        user_id: str,
    ):
        """Initialize the configured local store and processing pipeline."""
        self.cfg = cfg
        self.user_id = user_id
        self._megamem = AgentMemory(cfg, user_id=user_id)
        self._model_client = ChatCompletionModel(cfg)
        self.memory_builder_registry = MemoryBuilderRegistry()
        self.memory_builder_registry.register("markdown", DocumentMemoryBuilder)
        self.memory_builder_registry.register("doc", DocumentMemoryBuilder)
        self.memory_builder_registry.register("chat", ChatMemoryBuilder)
        self.memory_builder_registry.register("default", ChatMemoryBuilder)
        self.memory_builder_registry.register("email", EmailMemoryBuilder)

        self.processor_registry = ProcessorRegistry()
        self._source_cue_generator = SourceCueGenerator(cfg, self._model_client)

    def _resolve_builder(self, builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]], default_type: str = "default") -> MemoryBuilder:
        """
        Normalize the *builder* argument into a concrete MemoryBuilder instance.

        Args:
            builder: Can be a string (builder type), MemoryBuilder class, MemoryBuilder instance, or None
            default_type: Default builder type to use if builder is None

        Returns:
            MemoryBuilder instance
        """
        if isinstance(builder, MemoryBuilder):
            # A custom builder instance — use it directly.
            return builder
        if isinstance(builder, type) and issubclass(builder, MemoryBuilder):
            # A builder class — instantiate it.
            return builder(self.cfg, self._megamem, self._model_client)
        if isinstance(builder, str):
            # A type-name string — go through the registry.
            return self._get_memory_builder(builder)
        # Nothing usable — fall back to the registry's default type.
        return self._get_memory_builder(default_type)

    def _get_memory_builder(self, file_type: str) -> MemoryBuilder:
        """
        Pick the right memory builder for the given file type.

        Args:
            file_type: The detected file type (e.g., 'markdown', 'word', 'pdf')

        Returns:
            MemoryBuilder instance
        """
        # Mapping from file types to memory builder types.
        # Most file types currently route to the chat builder, but the table is extensible.
        # Emails are closer to documents structurally; a dedicated email builder
        # (decisions, attendees, ...) can be plugged in later.
        # Structured file processors emit text segments consumed by the document builder.
        builder_type_mapping = {
            "default": "chat",
            "chat": "chat",
            "markdown": "markdown",
            "word": "doc",
            "excel": "doc",
            "powerpoint": "doc",
            "pdf": "doc",
            "text": "doc",
            "richtext": "doc",
            "html": "doc",
            "json": "doc",
            "yaml": "doc",
            "toml": "doc",
            "config": "doc",
            "code": "doc",
            "xml": "doc",
            "email": "doc",
        }
        logger.info(f"Detected file type: {file_type}")
        builder_type = builder_type_mapping.get(file_type, "default")
        if builder_type not in self.memory_builder_registry._builders:
            logger.warning(
                f"file_type={file_type!r} -> builder_type={builder_type!r} not in "
                f"registry; falling back to 'default'. Registered: "
                f"{list(self.memory_builder_registry._builders.keys())}"
            )
            builder_type = "default"
        return self.memory_builder_registry.get(
            builder_type, self.cfg, self._megamem, self._model_client
        )

    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[MemoryEntry]:
        """Add memory content from a file; user_id is stamped automatically.

        Args:
            file_path: Path to the file to add.
            metadata: Additional metadata to store with the memory record
            builder: Optional custom builder. Can be:
                    - str: Builder type name (e.g., 'chat', 'markdown', 'doc')
                    - Type[MemoryBuilder]: MemoryBuilder class to instantiate
                    - MemoryBuilder: Custom MemoryBuilder instance
                    - None: Auto-detect based on file type (default)
            progress_callback: Optional callback for progress updates (current, total, message)

        Returns:
            Created or updated memory entries.
        """
        # Validate the file before doing real work.
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not is_supported_file_type(file_path):
            detected_type = detect_file_type(file_path)
            raise ValueError(
                f"Unsupported file type: {detected_type} (file: {file_path.name})"
            )

        segments = self._process_file(file_path)
        log_segments(segments)

        metadata = metadata or {}

        # Resolve the memory builder per the builder argument.
        if builder is None:
            # Auto-detect file type and pick the matching builder.
            file_type = detect_file_type(file_path)
            memory_builder = self._resolve_builder(builder, default_type=file_type)
        else:
            memory_builder = self._resolve_builder(builder)

        memory_entries = []
        surviving_indices: List[str] = []
        total_segments = len(segments)
        for pos, segment in enumerate(segments):
            if progress_callback:
                progress_callback(pos, total_segments, f"Processing segment {pos + 1}/{total_segments}")
            merged_metadata = merge_metadata(segment.metadata, metadata)
            results = memory_builder.build(segment.content, metadata=merged_metadata)
            memory_entries.extend(results)
            surviving_indices.extend(
                entry.index
                for entry in results
                if entry.is_primary_index() and entry.memory_type == "factual"
            )

        if memory_entries:
            source_metadata = dict(metadata)
            source_metadata.setdefault("data_type", "doc")
            source_metadata.setdefault(
                "title",
                source_metadata.get("file_title", "") or file_path.name,
            )
            source_metadata.setdefault(
                "author",
                source_metadata.get("file_creator", "")
                or source_metadata.get("sender", ""),
            )
            source_metadata.setdefault(
                "date",
                source_metadata.get("file_modified_time", "")
                or source_metadata.get("file_created_time", "")
                or source_metadata.get("timestamp", ""),
            )
            source_metadata.setdefault(
                "source_ref",
                source_metadata.get("file_url", "")
                or source_metadata.get("file_path", "")
                or str(file_path),
            )

            self._create_source_cue(
                memory_entries=memory_entries,
                content="",
                metadata=source_metadata,
                surviving_indices=surviving_indices,
            )

        if progress_callback:
            progress_callback(total_segments, total_segments, "All segments processed successfully")

        return memory_entries


    def _process_file(self, file_path: Union[str, Path]) -> List[Segment]:
        """
        Process a file into segments using the processor registry.

        Args:
            file_path: Path to the file to process

        Returns:
            List of Segment objects containing the processed content
        """
        file_path = Path(file_path)

        # Pick the processor for this file type.
        processor = self.processor_registry.get_processor(file_path)

        # Run the chosen processor over the file.
        return processor.process(file_path)

    def add(
        self,
        text: Union[str, List[str], List[Dict[str, str]]] = None,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,  # for progress bar
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
    ) -> List[MemoryEntry]:
        """Add memory content; user_id is stamped automatically.

        Args:
            text: Text to add. Can be:
                - str: Natural language text
                - List[str]: Multiple text entries
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with the memory record
            progress_callback: Optional callback for progress updates (current, total, message)
            builder: Optional custom builder. Can be:
                    - str: Builder type name (e.g., 'chat', 'markdown', 'doc')
                    - Type[MemoryBuilder]: MemoryBuilder class to instantiate
                    - MemoryBuilder: Custom MemoryBuilder instance
                    - None: Use default builder

        Returns:
            Record ID (derived from key)
        """
        if text is None:
            raise ValueError("Text must be provided")

        # Split text into segments so we can drive the progress callback.
        # Paragraph splits (double newlines) form natural chunk boundaries.
        if isinstance(text, str) and self.cfg.memory.enable_segmentation:
            # Prefer double-newline paragraphs, fall back to single newlines.
            raw_segments = text.split('\n\n') if '\n\n' in text else text.split('\n')
            # Drop empty segments and wrap the rest as Segments.
            segments = [
                Segment(content=chunk.strip(), segment_type="text", metadata=metadata)
                for chunk in raw_segments if chunk.strip()
            ]
            # If splitting produced nothing, fall back to a single segment.
            if not segments:
                segments = [Segment(content=text, segment_type="text", metadata=metadata)]
        else:
            # Non-string inputs are treated as a single segment.
            segments = [Segment(content=text, segment_type="text", metadata=metadata)]

        # Resolve the memory builder per the builder argument.
        memory_builder: MemoryBuilder = self._resolve_builder(builder, default_type="default")

        memory_entries = []
        total_segments = len(segments)
        for pos, segment in enumerate(segments):
            if progress_callback:
                progress_callback(pos, total_segments, f"Processing segment {pos + 1}/{total_segments}")
            merged_metadata = merge_metadata(segment.metadata, metadata)
            memory_entries.extend(
                memory_builder.build(segment.content, metadata=merged_metadata)
            )

        if progress_callback:
            progress_callback(total_segments, total_segments, "All segments processed successfully")

        return memory_entries

    def add_emails(
        self,
        emails: List[NormalizedEmail],
    ) -> List[MemoryEntry]:
        """Build memories from an email thread (or single email).

        Runs a thread-level filter that skips low-substance threads, then
        extracts memories from each email and creates a source cue per email
        carrying full metadata (sender, subject, recipients, date).

        Args:
            emails: List of emails in the thread (chronological).

        Returns:
            List of MemoryEntry objects created.
        """
        if not emails:
            return []

        builder = self.memory_builder_registry.get(
            "email", self.cfg, self._megamem, self._model_client
        )

        # Thread-level filter — skip low-substance threads.
        if not builder.should_process_thread(emails):
            return []

        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)

        all_entries: List[MemoryEntry] = []
        for email in emails:
            # Pull memories out of this email.
            entries, surviving_indices = builder._build_single_email(email, enable_episodic)
            if not entries:
                continue
            all_entries.extend(entries)

            # Build the metadata dict from the NormalizedEmail fields.
            sender_str = (
                f"{email.sender_name} <{email.sender_address}>"
                if email.sender_name
                else email.sender_address
            )
            recipients_str = ", ".join(
                r.get("name", r.get("address", ""))
                for r in email.to_recipients
            )
            email_metadata = {
                "data_type": "mail",
                "sender": sender_str,
                "subject": email.subject,
                "recipients": recipients_str,
                "date": email.sent_datetime,
                "source_ref": email.message_id,
            }

            # Create the source cue, link it to this email's primary memories,
            # and back-link the primary memories to the source cue.
            self._create_source_cue(
                memory_entries=entries,
                content="",  # not needed — data_type already set
                metadata=email_metadata,
                surviving_indices=surviving_indices,
            )

        return all_entries

    def add_chats(
        self,
        messages: List[NormalizedChatMessage],
    ) -> List[MemoryEntry]:
        """Build memories from a chat/Teams thread.

        Concatenates the messages into a single conversation, runs the chat
        builder, and creates a source cue with full metadata (topic, sender,
        recipients, dates).

        Mirrors the structure of :meth:`add_emails`.

        Args:
            messages: List of chat messages in a single thread (chronological).

        Returns:
            List of MemoryEntry objects created.
        """
        if not messages:
            return []

        # TeamsChatProcessor segments the messages into chunks.
        processor = TeamsChatProcessor(
            max_tokens_per_segment=self.cfg.memory.get("max_tokens_per_segment", 0),
        )
        segments = processor.process_messages(messages)

        if not segments:
            return []

        # Use the chat memory builder to extract memories per segment.
        memory_builder: MemoryBuilder = self._resolve_builder(None, default_type="chat")

        all_entries: List[MemoryEntry] = []

        for segment in segments:
            # Stamp the segment metadata with the source-cue data_type.
            seg_meta = {**segment.metadata, "data_type": "teams"}

            entries = memory_builder.build(segment.content, metadata=seg_meta)
            if not entries:
                continue

            all_entries.extend(entries)

            # Surviving indices for the source cue links.
            surviving_indices = [e.index for e in entries if e.is_primary_index()]

            # Create the source cue linking back to this segment's primary memories.
            self._create_source_cue(
                memory_entries=entries,
                content="",
                metadata=seg_meta,
                surviving_indices=surviving_indices,
            )

        return all_entries

    def planner_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        """
        Planner-driven retrieval pipeline for source-aware queries.

        Uses FILTER, RESOLVE, and SEMANTIC_SEARCH primitives to support
        sender/recipient filtering, document identification, and
        hierarchical drill-down.

        For simple semantic search, use query() instead.

        Args:
            context: Search context (str, List[str], or structured context)
            top_k: Maximum results to return
            latency_tracker: Optional latency tracker

        Returns:
            List of MemoryEntry objects
        """
        return self._megamem.planner_query(
            context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        query_mode: Optional[QueryMode] = None,
        **kwargs,
    ):
        """
        Vector search to surface memories matching the supplied context.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results (e.g., ["metadatas", "distances"])
            enable_hybrid_search: Whether to enable hybrid search combining semantic + keyword search
            enable_llm_filter: Whether to use LLM to filter irrelevant memories
            query_mode: Query mode (ORIGINAL, PRIMARY_ONLY, CUE_ONLY, or BOTH)

        Returns:
            Backend-specific result object containing matching memories
        """
        # Default the query mode from config when not explicitly set.
        if query_mode is None:
            query_mode = (
                QueryMode.BOTH
                if self.cfg.memory.enable_cue_index
                else QueryMode.PRIMARY_ONLY
            )

        return self._megamem.query(
            context,
            top_k=top_k,
            where=where,
            query_mode=query_mode,
            include=include,
            enhance_query=self.cfg.memory.enhance_query,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            **kwargs,  # Forward extras like latency_tracker.
        )

    def expand_by_session(
        self,
        memory_results: List[MemoryEntry],
        max_per_session: int = 5,
    ) -> List[MemoryEntry]:
        """Forward to AgentMemory.expand_by_session."""
        return self._megamem.expand_by_session(
            memory_results, max_per_session=max_per_session,
        )

    def get_all_cues(self) -> List[MemoryEntry]:
        """Forward to AgentMemory.get_all_cues."""
        return self._megamem.get_all_cues()

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        List every memory record for this user.

        Args:
            limit: Maximum number of records to return
        Returns:
            List of memory records as dictionaries
        """
        return self._megamem.list_memories(limit=limit)

    def get_user_id(self) -> str:
        return self.user_id

    def get(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record using its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        return self._megamem.get(key)

    def delete(self, key: str) -> None:
        """
        Remove a record using its natural-language key.

        Args:
            key: Natural language key to delete
        """
        self._megamem.delete(key)

    def count(self) -> int:
        """
        Return the total number of memory records stored.

        Returns:
            Total count of memory records
        """
        return self._megamem.count()

    def clear(self) -> None:
        """
        Drop every record in the collection.
        """
        self._megamem.clear()

    def delete_all(self, **kwargs) -> None:
        """
        Delete every record matching *param* in the collection.
        """

        if kwargs is None:
            param = {}
        else:
            param = {key: value for key, value in kwargs.items() if value is not None}

        self._megamem.delete_all(param)

    # ------------------------------------------------------------------
    # Source cue helpers (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_data_type(text: str) -> str:
        """
        Heuristic check for whether a chunk of text reads as an email or a document.

        Inspects the leading 500 chars for email-style headers (From:, To:, Subject:, etc.).
        Two or more such headers => "mail"; otherwise "doc".

        Args:
            text: Raw content text

        Returns:
            "mail" or "doc"
        """
        text_lower = text[:500].lower()
        email_signals = ["from:", "to:", "subject:", "sent:", "cc:", "bcc:"]
        matches = sum(1 for signal in email_signals if signal in text_lower)
        return "mail" if matches >= 2 else "doc"

    @staticmethod
    def _iso_to_unix(iso_str: str) -> int:
        """
        Convert an ISO timestamp string to a Unix timestamp (seconds).

        Handles formats like:
          - "2026-02-10T09:30:00Z"
          - "2026-02-10T09:30:00+00:00"
          - "2026-02-10" (date-only, midnight)

        Args:
            iso_str: ISO 8601 date/datetime string

        Returns:
            Unix timestamp as int, or 0 if parsing fails.
        """
        try:
            if iso_str.endswith('Z'):
                iso_str = iso_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            # Fall back to date-only parsing.
            try:
                dt = datetime.strptime(iso_str, "%Y-%m-%d")
                return int(dt.timestamp())
            except (ValueError, AttributeError):
                return 0

    def _create_source_cue(
        self,
        memory_entries: List[MemoryEntry],
        content: str,
        metadata: Optional[Dict] = None,
        surviving_indices: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Create a source-cue index linking every primary memory from a single source.

        Called after memory extraction in add() and add_file(). Uses the
        SourceCueGenerator to produce a natural-language source description
        from structured metadata only (no body content), then persists via
        AgentMemory.add_source_cue().

        Args:
            memory_entries: List of MemoryEntry objects created from this source
            content: Raw content text — used ONLY for auto-detecting data_type
                when not provided in metadata. NOT used for description generation.
            metadata: Caller-supplied metadata dict, may contain:
                - data_type: "mail" / "doc" / etc.
                - sender / author / subject / title / date: source-specific fields
                - source_ref: path or URI to the original source

        Returns:
            Record ID of the source cue entry, or None if skipped.
        """
        metadata = metadata or {}

        # Auto-detect data_type from the content headers when not supplied.
        data_type = metadata.get("data_type", "")
        if not data_type and content:
            data_type = self._detect_data_type(content)
            logger.info(f"Auto-detected data_type: {data_type}")

        # Auto-derive timestamp_unix from a 'date' or 'timestamp' metadata field.
        timestamp_unix = metadata.get("timestamp_unix", 0)
        if not timestamp_unix:
            date_str = metadata.get("date", "") or metadata.get("timestamp", "")
            if date_str:
                timestamp_unix = self._iso_to_unix(date_str)
                logger.info(f"Auto-computed timestamp_unix: {timestamp_unix} from '{date_str}'")

        # Pick the primary indices that actually live in the store.
        # When surviving_indices is supplied (from upsert_memory_entry return values), we
        # use those — they track the actual stored index even after merges/dedup.
        # Otherwise we fall back to the original entry.index values.

        # Why: the memory builder may merge a freshly extracted fact into an
        # existing entry (deduplication). When that happens the returned
        # MemoryEntry still carries the *original* candidate index, but only
        # the *surviving* (merged-into) index actually exists in the store.
        # Using memory_entries directly would therefore find nothing in the
        # store, causing the source cue to be skipped entirely.
        # surviving_indices tracks the final live index after any merge/dedup,
        # so the source cue always links to real, queryable entries.
        if surviving_indices:
            primary_indices = list({
                idx for idx in surviving_indices
                if self._megamem.get(idx) is not None
            })
        else:
            # Fallback for callers that don't pass surviving_indices.
            primary_indices = [
                entry.index for entry in memory_entries
                if entry.is_primary_index()
                and entry.memory_type == "factual"
                and self._megamem.get(entry.index) is not None
            ]
        if not primary_indices:
            logger.info("No primary factual memories to link. Skipping source cue.")
            return None

        # Assemble the source metadata for the LLM prompt — metadata only, no body content.
        # Keys are filtered per data_type via the SOURCE_TYPE_METADATA_KEYS registry.
        from megamem.core.source_cue_generator import get_metadata_keys_for_type
        source_meta = {"data_type": data_type}
        for key in get_metadata_keys_for_type(data_type):
            if key in metadata and metadata[key]:
                source_meta[key] = metadata[key]

        # Generate the natural-language source description.
        source_description = self._source_cue_generator.generate_source_cue(source_meta)

        # Build the filterable metadata fields stored on the source cue.
        extra_metadata = self._extract_filterable_metadata(metadata, data_type)

        # Persist the source cue via AgentMemory.
        rid = self._megamem.add_source_cue(
            source_description=source_description,
            linked_memory_indices=primary_indices,
            data_type=data_type,
            timestamp_unix=timestamp_unix,
            extra_metadata=extra_metadata,
        )

        logger.info(
            f"Source cue created: '{source_description[:60]}...' "
            f"linking {len(primary_indices)} memories"
        )
        return rid

    # ------------------------------------------------------------------
    # Per-data-type filterable-metadata extraction registry
    # ------------------------------------------------------------------
    # Each entry: (output_key, [candidate_input_keys], normalizer_func)
    # normalizer_func: callable(str) → str, applied to the first non-empty match.
    # New data type? Add an entry here.

    _FILTERABLE_FIELDS_REGISTRY: Dict[str, list] = {
        "mail": [
            ("sender",     ["sender", "from", "author"],    None),  # uses _normalize_sender
            ("subject",    ["subject"],                      str.lower),
            ("recipients", ["to", "recipients"],             str.lower),
        ],
        "doc": [
            ("author", ["author", "sender"],                     str.lower),
            ("title",  ["title", "filename", "file_name"],       str.lower),
        ],
        "teams": [
            ("participants", ["participants", "members"],    lambda v: ", ".join(v).lower() if isinstance(v, list) else str(v).lower()),
            ("topic",        ["topic", "subject"],          str.lower),
            ("conversation_type", ["thread_type", "conversation_type"], str.lower),
        ],
    }

    @staticmethod
    def _extract_filterable_metadata(metadata: dict, data_type: str) -> dict:
        """
        Build the filterable metadata fields stored on a source cue.

        Drives off _FILTERABLE_FIELDS_REGISTRY to extract and normalize fields
        per data_type. Extensible: add a new data_type entry to the registry
        dict to support additional source types.

        Args:
            metadata: Caller-supplied metadata dict
            data_type: Source category ("mail", "doc", or "")

        Returns:
            Dict of normalized filterable fields
        """
        filterable: Dict[str, Any] = {}

        field_specs = LocalMemoryClient._FILTERABLE_FIELDS_REGISTRY.get(data_type, [])
        for output_key, candidate_keys, normalizer in field_specs:
            # Pick the first non-empty candidate value.
            raw_value = ""
            for candidate in candidate_keys:
                raw_value = metadata.get(candidate, "")
                if raw_value:
                    break
            if not raw_value:
                continue

            # Apply the normalizer — sender gets bespoke handling.
            if output_key == "sender":
                filterable[output_key] = LocalMemoryClient._normalize_sender(raw_value)
            elif normalizer:
                filterable[output_key] = normalizer(raw_value)
            else:
                filterable[output_key] = raw_value

        # source_ref is shared across every data type.
        source_ref = (
            metadata.get("source_ref", "")
            or metadata.get("file_path", "")
            or metadata.get("message_id", "")
        )
        if source_ref:
            filterable["source_ref"] = source_ref

        return filterable

    @staticmethod
    def _normalize_sender(raw_sender: str) -> str:
        """
        Normalize a sender string for consistent $contains matching.

        Handles common formats:
          "Sarah Johnson <sarah.johnson@corp.com>" → "sarah johnson"
          "sarah.johnson@corp.com"                 → "sarah johnson"
          "Sarah J."                               → "sarah j."

        Args:
            raw_sender: Raw sender string from email metadata

        Returns:
            Lowercased, cleaned sender name
        """
        import re
        if not raw_sender:
            return ""

        # "Name <email>" → keep "Name".
        name_match = re.match(r'^([^<]+)<', raw_sender)
        if name_match:
            name = name_match.group(1).strip()
        elif '@' in raw_sender:
            # Pure email: take the local part, swap dots/underscores for spaces.
            local_part = raw_sender.split('@')[0]
            name = re.sub(r'[._]', ' ', local_part)
        else:
            name = raw_sender

        return name.lower().strip()
