from dataclasses import dataclass
from statistics import mean
from typing import List, Set, Tuple

from .trajectory_utils import Trajectory


@dataclass
class TrajectoryScore:
    groundedness: float
    redundancy: float
    cost: float
    total_score: float


class TrajectoryScorer:

    def __init__(
        self,
        w_groundedness: float = 1.0,
        w_redundancy: float = 0.3,
        w_cost: float = 0.1,
        redundancy_threshold: float = 0.5,
    ):
        self.w1 = w_groundedness
        self.w2 = w_redundancy
        self.w3 = w_cost
        self.redundancy_threshold = redundancy_threshold

    def compute_groundedness(
        self,
        trajectory: Trajectory,
    ) -> float:
        retrieved_ids: Set[str] = set()
        retrieved_text: List[str] = []
        for memory in trajectory.retrieved_memories:
            for key in ("id", "index", "node_id", "source_evidence_id"):
                value = memory.get(key)
                if value:
                    retrieved_ids.add(str(value))
            for value in memory.get("source_evidence_ids", []) or []:
                retrieved_ids.add(str(value))
            text = memory.get("value") or memory.get("content") or memory.get("document")
            if text:
                retrieved_text.append(str(text))

        expected_ids = {str(value) for value in trajectory.evidence if value}
        if expected_ids:
            return len(expected_ids & retrieved_ids) / len(expected_ids)

        answer_tokens = self._tokens(trajectory.ground_truth or "")
        if not answer_tokens:
            return 0.0
        memory_tokens = self._tokens(" ".join(retrieved_text))
        return len(answer_tokens & memory_tokens) / len(answer_tokens)

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {
            "".join(char for char in token.lower() if char.isalnum())
            for token in text.split()
            if any(char.isalnum() for char in token)
        }

    def compute_redundancy(
        self,
        trajectory: Trajectory,
    ) -> float:
        memories = trajectory.retrieved_memories
        n = len(memories)

        if n <= 1:
            return 0.0

        total_pairs = n * (n - 1) / 2
        redundant_pairs = 0

        for i in range(n):
            text_i = memories[i].get("value", "")
            for j in range(i + 1, n):
                if self._text_similarity(text_i, memories[j].get("value", "")) > self.redundancy_threshold:
                    redundant_pairs += 1

        return redundant_pairs / max(total_pairs, 1)

    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        union = len(words1 | words2)
        if union == 0:
            return 0.0
        return len(words1 & words2) / union

    def compute_cost(self, trajectory: Trajectory) -> float:
        if trajectory.cost is None:
            return 0.0
        return min(trajectory.cost / 10.0, 1.0)

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        groundedness = self.compute_groundedness(trajectory)
        redundancy = self.compute_redundancy(trajectory)
        cost = self.compute_cost(trajectory)

        total = (
            self.w1 * groundedness
            - self.w2 * redundancy
            - self.w3 * cost
        )

        return TrajectoryScore(
            groundedness=groundedness,
            redundancy=redundancy,
            cost=cost,
            total_score=total,
        )

    def compute_group_advantages(
        self,
        trajectories: List[Trajectory],
    ) -> List[float]:
        scores = [self.score_trajectory(t).total_score for t in trajectories]
        mean_score = mean(scores) if scores else 0.0
        return [s - mean_score for s in scores]


def score_trajectory_batch(
    trajectories: List[Trajectory],
    scorer: TrajectoryScorer = None,
) -> Tuple[List[TrajectoryScore], List[float]]:
    if scorer is None:
        scorer = TrajectoryScorer()

    scores = [scorer.score_trajectory(t) for t in trajectories]
    advantages = scorer.compute_group_advantages(trajectories)

    return scores, advantages
