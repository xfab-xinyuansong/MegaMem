import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from megamem.core.base import MemoryStoreBase
from megamem.core.memory_entry import MemoryEntry
from megamem.utils.misc import get_current_timestamp, index_to_id

logger = logging.getLogger(__name__)


def merge_with_rrf(
    result_lists: List[List[MemoryEntry]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[MemoryEntry]:
    """Fuse ranked memory lists using weighted Reciprocal Rank Fusion.

    For every entry ``d``, the RRF score is::

        RRF_score(d) = sum_i weight_i / (k + rank_i(d))

    summed over the lists in which ``d`` appears. Scores are min-max
    normalised to ``[0, 1]`` and stored on each entry's ``score`` field;
    the returned list is sorted from highest to lowest score.

    Args:
        result_lists: ranked candidate lists from different retrievers.
        weights: per-list weights; defaults to ``1.0`` for every list.
        k: smoothing constant in the RRF formula (default 60).
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) != result_lists length ({len(result_lists)})"
        )

    fused_scores: dict[str, float] = {}
    seen_entries: dict[str, MemoryEntry] = {}

    for ranked, w in zip(result_lists, weights):
        for rank, entry in enumerate(ranked, start=1):
            rec_id = index_to_id(entry.index)
            if rec_id not in seen_entries:
                seen_entries[rec_id] = entry
            fused_scores[rec_id] = fused_scores.get(rec_id, 0.0) + w / (k + rank)

    if fused_scores:
        hi = max(fused_scores.values())
        lo = min(fused_scores.values())
        spread = hi - lo
        if spread > 0:
            fused_scores = {
                rid: (s - lo) / spread for rid, s in fused_scores.items()
            }
        else:
            fused_scores = {rid: 1.0 for rid in fused_scores}

    for rec_id, score in fused_scores.items():
        if rec_id in seen_entries:
            seen_entries[rec_id].score = score

    return sorted(seen_entries.values(), key=lambda x: x.score, reverse=True)


class MemoryUpdateDecision(BaseModel):
    """Structured LLM output describing whether to update an existing memory.

    Using a Pydantic schema instead of free-form JSON gives us validation
    and removes brittle manual parsing in the call sites.

    Attributes:
        should_update: whether an existing memory should be updated.
        best_candidate_index: chosen candidate index when ``should_update``.
        updated_index: rewritten index string for the merged memory.
        updated_cue_indices: cue indices to attach to the merged memory.
    """

    should_update: bool = Field(
        description="Whether an existing memory entry should be updated"
    )
    best_candidate_index: Optional[int] = Field(
        description="Index of the best candidate to update if should_update is True",
        default=None,
    )
    updated_index: Optional[str] = Field(
        description="Updated index if needed, or original index", default=None
    )
    updated_cue_indices: List[str] = Field(
        description="Updated list of 1-3 cue indices for the merged memory. Each cue is a 2-4 word phrase following [Main Entity] + [Key Aspect] pattern. Generate fresh cue indices that cover diverse aspects of the merged memory content.",
        default_factory=list,
    )
    # Note: reasoning field commented out to reduce LLM response complexity
    # reasoning: str = Field(description="Explanation of the decision")


def combine_list(list_string1: str, list_string2: str, delimiter: str = "||") -> str:
    """Merge two delimited "list-as-string" values, removing duplicates.

    Empty pieces and exact duplicates are dropped while order is otherwise
    irrelevant (a ``set`` is used internally).

    Args:
        list_string1: first delimited list (may be empty).
        list_string2: second delimited list (may be empty).
        delimiter: token used to split/join the two strings (default "||").

    Returns:
        The merged delimited string, or "" when both inputs are empty.

    Examples:
        combine_list("item1", "item2") -> "item1||item2"
        combine_list("item1||item2", "item3") -> "item1||item2||item3"
        combine_list("", "item1") -> "item1"
        combine_list("item1", "") -> "item1"
        combine_list("item1||item2", "item1||item3") -> "item1||item2||item3"
    """
    if not list_string1 and not list_string2:
        return ""
    if not list_string1:
        return list_string2
    if not list_string2:
        return list_string1

    if list_string1 == list_string2:
        return list_string1

    parts1 = [p.strip() for p in list_string1.split(delimiter) if p.strip()]
    parts2 = [p.strip() for p in list_string2.split(delimiter) if p.strip()]

    merged_unique = list(set(parts1 + parts2))

    return delimiter.join(merged_unique)


def generate_metadata(content: str, metadata: Optional[dict]) -> dict:

    if not metadata:
        metadata = {}

    metadata["creation_time"] = get_current_timestamp()

    if isinstance(content, dict) and "image" in content:
        urls = []
        for piece in content["image"]:
            if piece.get("type") == "image_url":
                u = piece.get("image_url", {}).get("url")
                if u:
                    urls.append(u)
        if urls:
            metadata["image_urls"] = urls
    return metadata


def delete_candidate_memory(candidate: MemoryEntry, memory_store: MemoryStoreBase):
    """Remove a candidate memory and any cue-index links pointing at it.

    Args:
        candidate: the memory entry to delete.
        memory_store: backing store on which deletes are issued.
    """
    cand_idx = candidate.index
    cues: str = candidate.cue_indices

    memory_store.delete(cand_idx)

    if cues:
        for cue in cues.split("||"):
            existing = memory_store.get(cue)
            if not existing:
                raise ValueError(
                    f"Index entry '{cue}' not found in memory store. Please check."
                )
            memory_store.delete(cue)


def generate_history(entry: MemoryEntry, best_candidate: MemoryEntry) -> List[dict]:
    if best_candidate.history:
        history = best_candidate.history
    else:
        history = [
            {
                "index": best_candidate.index,
                "value": best_candidate.value,
                "creation_time": best_candidate.creation_time,
                "timestamp": best_candidate.timestamp,
            }
        ]

    history += [
        {
            "index": entry.index,
            "value": entry.value,
            "creation_time": entry.creation_time,
            "timestamp": entry.timestamp,
        }
    ]

    return history


def convert_memory_output(
    memories: Any, metadata: dict, enable_cue_index: bool
) -> List[MemoryEntry]:
    out_entries: List[MemoryEntry] = []
    epi_id = metadata.get("episodic_memory_id", None)

    for raw in memories.entries:
        cues = ""
        if (
            enable_cue_index
            and hasattr(raw, "cue_indices")
            and raw.cue_indices
        ):
            cues = "||".join(raw.cue_indices)

        epi_ids = [epi_id] if epi_id else []

        out_entries.append(
            MemoryEntry(
                memory_type=raw.memory_type,
                index=raw.index,
                value=raw.value,
                creation_time=metadata["creation_time"],
                timestamp=metadata.get("timestamp", ""),
                cue_indices=cues,
                episodic_memory_ids=epi_ids,
            )
        )
    return out_entries


def format_memories_to_str(
    memories: List[MemoryEntry],
    enable_episodic: bool = False,
) -> str:
    """Render a list of ``MemoryEntry`` objects to a printable string.

    Args:
        memories: entries to render.
        enable_episodic: when ``True`` group entries by their associated
            episodic ids; when ``False`` use the simple ``timestamp: value``
            line-per-entry format.

    Returns:
        Formatted text representation. Empty string for an empty input.
    """
    if not memories:
        return ""

    if not enable_episodic:
        lines = []
        for mem in memories:
            ts = mem.timestamp if mem.timestamp else ""
            lines.append(f"{ts}: {mem.value}")
        return "\n".join(lines)

    # Episodic clustering branch.
    clusters: dict = {}
    standalone: List[MemoryEntry] = []

    for mem in memories:
        if mem.episodic_memory_ids and len(mem.episodic_memory_ids) > 0:
            key = tuple(mem.episodic_memory_ids)
            clusters.setdefault(key, []).append(mem)
        else:
            standalone.append(mem)

    lines: List[str] = []

    for epi_ids, cluster in clusters.items():
        cluster.sort(key=lambda m: m.score or 0, reverse=True)

        lines.append(f"Related to episodes: {', '.join(epi_ids)}")
        lines.append("Details:")
        for mem in cluster:
            ts = mem.timestamp if mem.timestamp else ""
            lines.append(f"{ts}: {mem.value}")
        lines.append("")

    for mem in standalone:
        ts = mem.timestamp if mem.timestamp else ""
        lines.append(f"{ts}: {mem.value}")

    return "\n".join(lines)


def dedup_memories(memories: List[MemoryEntry]) -> List[MemoryEntry]:
    """Deduplicate memories by their ``index``, keeping the first occurrence.

    Entries without an ``index`` are kept verbatim (they normally shouldn't
    occur in well-formed data).

    Args:
        memories: possibly duplicated entries.

    Returns:
        List with duplicates removed in original order.

    Example:
        memories = [entry1, entry2, entry1, entry3]  # entry1 appears twice
        unique_memories = dedup(memories)  # Returns [entry1, entry2, entry3]
    """
    seen: set = set()
    unique: List[MemoryEntry] = []

    for mem in memories:
        if mem.index and mem.index not in seen:
            seen.add(mem.index)
            unique.append(mem)
        elif not mem.index:
            unique.append(mem)

    return unique
