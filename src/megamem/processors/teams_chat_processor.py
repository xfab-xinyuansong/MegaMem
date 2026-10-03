import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from megamem.builder.chat_memory_builder import NormalizedChatMessage
from megamem.core.segment import Segment
from megamem.processors.base_processor import BaseProcessor


class TeamsChatProcessor(BaseProcessor):

    def __init__(self, max_tokens_per_segment: int = 0) -> None:
        self.max_tokens_per_segment = max_tokens_per_segment

    def can_process(self, file_path: Path) -> bool:
        if file_path.suffix.lower() != ".json":
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return False
        return (
            isinstance(payload, dict)
            and "cache_info" in payload
            and payload.get("cache_info", {}).get("data_type") == "teams"
        )

    def process(self, file_path: Path) -> List[Segment]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        cache_info: Dict[str, Any] = raw.get("cache_info", {})
        messages: List[Dict[str, Any]] = raw.get("data", [])

        return self._process_raw_messages(messages, cache_info=cache_info, source_file=str(file_path))

    def process_messages(
        self,
        messages: List[NormalizedChatMessage],
        *,
        source_label: str = "teams_chat",
    ) -> List[Segment]:
        raw_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            d = asdict(msg)
            d.setdefault("received_datetime", d.get("sent_datetime", ""))
            d.setdefault("body_content", d.get("body_text", ""))
            raw_msgs.append(d)

        return self._process_raw_messages(
            raw_msgs, source_file=source_label,
        )

    def _process_raw_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        cache_info: Optional[Dict[str, Any]] = None,
        source_file: str = "",
    ) -> List[Segment]:
        if cache_info is None:
            cache_info = {}

        threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for msg in messages:
            tid = msg.get("thread_id") or msg.get("conversation_id") or "unknown"
            threads[tid].append(msg)

        segments: List[Segment] = []
        for thread_id, thread_msgs in threads.items():
            thread_msgs.sort(key=lambda m: m.get("received_datetime", ""))

            head = thread_msgs[0]
            topic = head.get("topic", "") or head.get("subject", "")
            thread_type = head.get("thread_type", "chat")

            base_metadata = self._build_thread_metadata(
                Path(source_file), cache_info, thread_id, topic, thread_type, thread_msgs,
            )

            formatted = self._format_thread(thread_msgs)

            if self.max_tokens_per_segment > 0:
                chunks = self._chunk_text(formatted, self.max_tokens_per_segment)
                total_chunks = len(chunks)
                for chunk_idx, chunk in enumerate(chunks):
                    segments.append(
                        Segment(
                            content=chunk,
                            segment_type="teams_thread",
                            metadata={
                                **base_metadata,
                                "thread_chunk_index": chunk_idx + 1,
                                "thread_chunk_count": total_chunks,
                            },
                        )
                    )
            else:
                segments.append(
                    Segment(
                        content=formatted,
                        segment_type="teams_thread",
                        metadata=base_metadata,
                    )
                )

        return segments

    @staticmethod
    def _format_message(msg: Dict[str, Any]) -> str:
        sender = msg.get("sender_name", "Unknown")
        timestamp = msg.get("received_datetime", "")
        body = (msg.get("body_content") or "").strip()
        return f"[{timestamp}] {sender}: {body}"

    def _format_thread(self, msgs: List[Dict[str, Any]]) -> str:
        head = msgs[0]
        topic = head.get("topic", "") or head.get("subject", "")
        header = f"Thread: {topic}\n" if topic else ""
        lines = [self._format_message(m) for m in msgs]
        return header + "\n".join(lines)

    @staticmethod
    def _chunk_text(text: str, max_tokens: int) -> List[str]:
        words = text.split()
        if len(words) <= max_tokens:
            return [text]
        return [
            " ".join(words[i : i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]

    @staticmethod
    def _build_thread_metadata(
        file_path: Path,
        cache_info: Dict[str, Any],
        thread_id: str,
        topic: str,
        thread_type: str,
        thread_msgs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        participants = {m.get("sender_name", "Unknown") for m in thread_msgs}
        first = thread_msgs[0]
        last = thread_msgs[-1]
        return {
            "source_file": str(file_path),
            "file_type": "teams_chat",
            "file_name": file_path.name,
            "user_email": cache_info.get("user_email", ""),
            "thread_id": thread_id,
            "topic": topic,
            "thread_type": thread_type,
            "message_count": len(thread_msgs),
            "participants": sorted(participants),
            "start_datetime": first.get("received_datetime", ""),
            "end_datetime": last.get("received_datetime", ""),
        }
