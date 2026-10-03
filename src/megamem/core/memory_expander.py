from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from megamem.core.memory_entry import MemoryEntry
from megamem.core.memory import AgentMemory, QueryMode


class MemoryExpander:

    def __init__(self, memory_client: Optional[AgentMemory] = None,
                 enable_relaxed_frontier: bool = False,
                 relaxed_frontier_top_k: int = 4,
                 relaxed_frontier_threshold: float = 0.85,
                 max_cues_to_expand: int = 30,
                 max_workers: int = 5):
        self.visited_ids: Set[str] = set()
        self.memory_client = memory_client
        self.enable_relaxed_frontier = enable_relaxed_frontier
        self.relaxed_frontier_top_k = relaxed_frontier_top_k
        self.relaxed_frontier_threshold = relaxed_frontier_threshold
        self.max_cues_to_expand = max_cues_to_expand
        self.max_workers = max_workers

    def set_memory_client(self, memory_client: AgentMemory):
        self.memory_client = memory_client

    def build_frontier(
            self,
            frontier: Dict[str, MemoryEntry],
            memories: List[MemoryEntry]
    ) -> Dict[str, MemoryEntry]:

        if self.memory_client is None:
            raise ValueError("memory_client must be set before calling build_frontier()")

        print("=="*40)
        print('\n')
        print("Building frontier")
        print("=="*40)

        working_set = {m.index for m in memories}

        cue_to_memory_score: Dict[str, float] = {}
        direct_cues: Set[str] = set()

        for mem in memories:
            if mem.index in self.visited_ids:
                continue
            self.visited_ids.add(mem.index)

            mem_score = mem.score if mem.score is not None else 1.0

            for cue_index in mem.get_cue_indices():
                if cue_index in self.visited_ids:
                    continue
                direct_cues.add(cue_index)
                current_best = cue_to_memory_score.get(cue_index)
                if current_best is None or mem_score > current_best:
                    cue_to_memory_score[cue_index] = mem_score

        all_cues: Set[str] = set(direct_cues)

        if self.enable_relaxed_frontier and direct_cues:
            sorted_cues = sorted(cue_to_memory_score.items(), key=lambda pair: pair[1], reverse=True)
            cues_to_expand = [cue for cue, _ in sorted_cues[:self.max_cues_to_expand]]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_cue = {
                    executor.submit(self._find_similar_cues, cue_index): cue_index
                    for cue_index in cues_to_expand
                }

                for future in as_completed(future_to_cue):
                    similar_cues = future.result()
                    for similar_cue in similar_cues:
                        if similar_cue.index not in self.visited_ids:
                            all_cues.add(similar_cue.index)

        for cue_id in all_cues:
            if cue_id in self.visited_ids:
                continue
            self.visited_ids.add(cue_id)

            cue_entry = self.memory_client.get(cue_id)
            if not cue_entry:
                continue

            for linked_index in cue_entry.get_linked_memories():
                if linked_index in working_set or linked_index in frontier:
                    continue

                linked_entry = self.memory_client.get(linked_index)
                if linked_entry:
                    frontier[linked_entry.index] = linked_entry

        return frontier

    def _find_similar_cues(self, cue_index: str) -> List[MemoryEntry]:
        try:
            similar_cues = self.memory_client.query(
                cue_index,
                top_k=self.relaxed_frontier_top_k,
                enable_hybrid_search=False,
                query_mode=QueryMode.CUE_ONLY
            )

            return [
                cue for cue in similar_cues
                if cue.score >= self.relaxed_frontier_threshold
                and cue.index != cue_index
            ]

        except Exception as e:
            print(f"Error finding similar cues for '{cue_index}': {e}")
            return []

    def reset(self):
        self.visited_ids.clear()
