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

    def __init__(
        self,
        cfg: DictConfig,
        user_id: str,
    ):
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
        if isinstance(builder, MemoryBuilder):
            return builder
        if isinstance(builder, type) and issubclass(builder, MemoryBuilder):
            return builder(self.cfg, self._megamem, self._model_client)
        if isinstance(builder, str):
            return self._get_memory_builder(builder)
        return self._get_memory_builder(default_type)

    def _get_memory_builder(self, file_type: str) -> MemoryBuilder:

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

        if builder is None:
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
        file_path = Path(file_path)

        processor = self.processor_registry.get_processor(file_path)

        return processor.process(file_path)

    def add(
        self,
        text: Union[str, List[str], List[Dict[str, str]]] = None,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
    ) -> List[MemoryEntry]:
        if text is None:
            raise ValueError("Text must be provided")

        if isinstance(text, str) and self.cfg.memory.enable_segmentation:
            raw_segments = text.split('\n\n') if '\n\n' in text else text.split('\n')
            segments = [
                Segment(content=chunk.strip(), segment_type="text", metadata=metadata)
                for chunk in raw_segments if chunk.strip()
            ]
            if not segments:
                segments = [Segment(content=text, segment_type="text", metadata=metadata)]
        else:
            segments = [Segment(content=text, segment_type="text", metadata=metadata)]

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
        if not emails:
            return []

        builder = self.memory_builder_registry.get(
            "email", self.cfg, self._megamem, self._model_client
        )

        if not builder.should_process_thread(emails):
            return []

        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)

        all_entries: List[MemoryEntry] = []
        for email in emails:
            entries, surviving_indices = builder._build_single_email(email, enable_episodic)
            if not entries:
                continue
            all_entries.extend(entries)

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

            self._create_source_cue(
                memory_entries=entries,
                content="",
                metadata=email_metadata,
                surviving_indices=surviving_indices,
            )

        return all_entries

    def add_chats(
        self,
        messages: List[NormalizedChatMessage],
    ) -> List[MemoryEntry]:
        if not messages:
            return []

        processor = TeamsChatProcessor(
            max_tokens_per_segment=self.cfg.memory.get("max_tokens_per_segment", 0),
        )
        segments = processor.process_messages(messages)

        if not segments:
            return []

        memory_builder: MemoryBuilder = self._resolve_builder(None, default_type="chat")

        all_entries: List[MemoryEntry] = []

        for segment in segments:
            seg_meta = {**segment.metadata, "data_type": "teams"}

            entries = memory_builder.build(segment.content, metadata=seg_meta)
            if not entries:
                continue

            all_entries.extend(entries)

            surviving_indices = [e.index for e in entries if e.is_primary_index()]

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
            **kwargs,
        )

    def expand_by_session(
        self,
        memory_results: List[MemoryEntry],
        max_per_session: int = 5,
    ) -> List[MemoryEntry]:
        return self._megamem.expand_by_session(
            memory_results, max_per_session=max_per_session,
        )

    def get_all_cues(self) -> List[MemoryEntry]:
        return self._megamem.get_all_cues()

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        return self._megamem.list_memories(limit=limit)

    def get_user_id(self) -> str:
        return self.user_id

    def get(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        return self._megamem.get(key)

    def delete(self, key: str) -> None:
        self._megamem.delete(key)

    def count(self) -> int:
        return self._megamem.count()

    def clear(self) -> None:
        self._megamem.clear()

    def delete_all(self, **kwargs) -> None:

        if kwargs is None:
            param = {}
        else:
            param = {key: value for key, value in kwargs.items() if value is not None}

        self._megamem.delete_all(param)

    @staticmethod
    def _detect_data_type(text: str) -> str:
        text_lower = text[:500].lower()
        email_signals = ["from:", "to:", "subject:", "sent:", "cc:", "bcc:"]
        matches = sum(1 for signal in email_signals if signal in text_lower)
        return "mail" if matches >= 2 else "doc"

    @staticmethod
    def _iso_to_unix(iso_str: str) -> int:
        try:
            if iso_str.endswith('Z'):
                iso_str = iso_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
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
        metadata = metadata or {}

        data_type = metadata.get("data_type", "")
        if not data_type and content:
            data_type = self._detect_data_type(content)
            logger.info(f"Auto-detected data_type: {data_type}")

        timestamp_unix = metadata.get("timestamp_unix", 0)
        if not timestamp_unix:
            date_str = metadata.get("date", "") or metadata.get("timestamp", "")
            if date_str:
                timestamp_unix = self._iso_to_unix(date_str)
                logger.info(f"Auto-computed timestamp_unix: {timestamp_unix} from '{date_str}'")

        if surviving_indices:
            primary_indices = list({
                idx for idx in surviving_indices
                if self._megamem.get(idx) is not None
            })
        else:
            primary_indices = [
                entry.index for entry in memory_entries
                if entry.is_primary_index()
                and entry.memory_type == "factual"
                and self._megamem.get(entry.index) is not None
            ]
        if not primary_indices:
            logger.info("No primary factual memories to link. Skipping source cue.")
            return None

        from megamem.core.source_cue_generator import get_metadata_keys_for_type
        source_meta = {"data_type": data_type}
        for key in get_metadata_keys_for_type(data_type):
            if key in metadata and metadata[key]:
                source_meta[key] = metadata[key]

        source_description = self._source_cue_generator.generate_source_cue(source_meta)

        extra_metadata = self._extract_filterable_metadata(metadata, data_type)

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

    _FILTERABLE_FIELDS_REGISTRY: Dict[str, list] = {
        "mail": [
            ("sender",     ["sender", "from", "author"],    None),
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
        filterable: Dict[str, Any] = {}

        field_specs = LocalMemoryClient._FILTERABLE_FIELDS_REGISTRY.get(data_type, [])
        for output_key, candidate_keys, normalizer in field_specs:
            raw_value = ""
            for candidate in candidate_keys:
                raw_value = metadata.get(candidate, "")
                if raw_value:
                    break
            if not raw_value:
                continue

            if output_key == "sender":
                filterable[output_key] = LocalMemoryClient._normalize_sender(raw_value)
            elif normalizer:
                filterable[output_key] = normalizer(raw_value)
            else:
                filterable[output_key] = raw_value

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
        import re
        if not raw_sender:
            return ""

        name_match = re.match(r'^([^<]+)<', raw_sender)
        if name_match:
            name = name_match.group(1).strip()
        elif '@' in raw_sender:
            local_part = raw_sender.split('@')[0]
            name = re.sub(r'[._]', ' ', local_part)
        else:
            name = raw_sender

        return name.lower().strip()
