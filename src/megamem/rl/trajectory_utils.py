from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from omegaconf import DictConfig
import random
import math
import logging

if TYPE_CHECKING:
    from megamem.client import MemoryClient
    from megamem.core.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


class ActionType(Enum):
    QUERY_PRIMARY_INDEX = "query_primary"
    QUERY_CUE_INDEX = "query_cue"

    STOP = "stop"


ACTION_COSTS = {
    ActionType.QUERY_PRIMARY_INDEX: 1.0,
    ActionType.QUERY_CUE_INDEX: 1.0,

    ActionType.STOP: 0.0,
}


@dataclass
class RetrievalState:
    query: str
    retrieved_memories: List[str]
    frontier: List[str]
    budget: float
    step: int = 0


@dataclass
class RetrievalAction:
    action_type: ActionType
    target_memory_index: Optional[str] = None
    new_query: Optional[str] = None
    score: float = 0.0


@dataclass
class RetrievalStep:
    step_idx: int
    state: Dict
    action: Dict
    reward: float = 0.0


@dataclass
class Trajectory:
    query: str
    user_id: str
    steps: List[RetrievalStep] = field(default_factory=list)
    retrieved_memories: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    trajectory_score: Optional[float] = None

    groundedness: Optional[float] = None
    redundancy: Optional[float] = None
    cost: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "steps": [asdict(step) for step in self.steps],
            "retrieved_memories": self.retrieved_memories,
            "final_answer": self.final_answer,
            "ground_truth": self.ground_truth,
            "evidence": self.evidence,
            "trajectory_score": self.trajectory_score,
            "groundedness": self.groundedness,
            "redundancy": self.redundancy,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        return cls(
            query=data["query"],
            user_id=data["user_id"],
            steps=[RetrievalStep(**s) for s in data.get("steps", [])],
            retrieved_memories=data.get("retrieved_memories", []),
            final_answer=data.get("final_answer"),
            ground_truth=data.get("ground_truth"),
            evidence=data.get("evidence", []),
            trajectory_score=data.get("trajectory_score"),
            groundedness=data.get("groundedness"),
            redundancy=data.get("redundancy"),
            cost=data.get("cost"),
        )


class TrajectoryCollector:

    def __init__(
            self,
            cfg: DictConfig,
            top_k: int = 20,
            budget: float = 10.0,
            max_steps: int = 15,
    ):
        self.cfg = cfg
        self.top_k = top_k
        self.initial_budget = budget
        self.max_steps = max_steps
        self.client_cache: Dict[str, MemoryClient] = {}

    def get_client(self, user_id: str) -> MemoryClient:
        from megamem.client import MemoryClient

        client = self.client_cache.get(user_id)
        if client is None:
            client = MemoryClient(self.cfg, user_id=user_id)
            self.client_cache[user_id] = client
        return client

    def _init_frontier(
            self,
            query: str,
            client: MemoryClient,
    ) -> Tuple[List[MemoryEntry], List[MemoryEntry]]:
        primary_results = client.query(
            query,
            top_k=self.top_k,
            where={"memory_type": {"$eq": "factual"}},
        )

        cue_results = client.query(
            query,
            top_k=self.top_k // 2,
            where={"linked_memory": {"$ne": ""}},
        )

        return primary_results, cue_results

    def _select_action(
        self,
        state: RetrievalState,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
        temperature: float = 0.0,
        policy_network=None,
    ) -> RetrievalAction:

        if policy_network is not None:
            action = policy_network.select_action(
                state=state,
                primary_candidates=primary_candidates,
                cue_candidates=cue_candidates,
            )
            if not isinstance(action, RetrievalAction):
                raise TypeError("policy_network.select_action() must return RetrievalAction")
            return action

        already = state.retrieved_memories
        available_primary = [m for m in primary_candidates if m.index not in already]
        available_cue = [m for m in cue_candidates if m.index not in already]

        if state.budget <= 0 or (not available_primary and not available_cue):
            return RetrievalAction(action_type=ActionType.STOP)

        all_candidates: list = []
        all_candidates.extend((m, ActionType.QUERY_PRIMARY_INDEX, m.score) for m in available_primary)
        all_candidates.extend((m, ActionType.QUERY_CUE_INDEX, m.score) for m in available_cue)

        if not all_candidates:
            return RetrievalAction(action_type=ActionType.STOP)

        if temperature > 0 and len(all_candidates) > 1:
            scores = [c[2] for c in all_candidates]
            probs = self._softmax(scores, temperature)
            chosen_idx = random.choices(range(len(all_candidates)), weights=probs)[0]
        else:
            chosen_idx = max(range(len(all_candidates)), key=lambda i: all_candidates[i][2])

        chosen = all_candidates[chosen_idx]
        return RetrievalAction(
            action_type=chosen[1],
            target_memory_index=chosen[0].index,
            score=chosen[2],
        )

    def _apply_action(
        self,
        action: RetrievalAction,
        state: RetrievalState,
        client: MemoryClient,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> Tuple[Set[str], MemoryEntry, float]:
        new_memories: Set[str] = set()
        retrieved_entry: Optional[MemoryEntry] = None
        cost = ACTION_COSTS[action.action_type]

        if action.action_type == ActionType.STOP:
            return new_memories, retrieved_entry, cost

        if action.action_type in (ActionType.QUERY_PRIMARY_INDEX, ActionType.QUERY_CUE_INDEX):
            for m in primary_candidates + cue_candidates:
                if m.index == action.target_memory_index:
                    retrieved_entry = m
                    break

            if retrieved_entry is not None:
                if retrieved_entry.is_cue_index():
                    for primary_idx in retrieved_entry.get_linked_memories():
                        if primary_idx in state.retrieved_memories:
                            continue
                        primary_entry = client._client._megamem.get(primary_idx)
                        if primary_entry:
                            new_memories.add(primary_idx)
                else:
                    new_memories.add(retrieved_entry.index)

        return new_memories, retrieved_entry, cost

    def collect_single_trajectory(
            self,
            query: str,
            user_id: str,
            ground_truth: str = None,
            evidence: List[str] = None,
            temperature: float = 0.0,
            policy_network=None,
    ) -> Trajectory:
        client = self.get_client(user_id)

        state = RetrievalState(
            query=query,
            retrieved_memories=[],
            frontier=[],
            budget=self.initial_budget,
            step=0,
        )

        primary_candidates, cue_candidates = self._init_frontier(query, client)

        trajectory = Trajectory(
            query=query,
            user_id=user_id,
            ground_truth=ground_truth,
            evidence=evidence or [],
        )

        total_cost = 0.0

        for t in range(self.max_steps):
            state.step = t

            action = self._select_action(
                state,
                primary_candidates,
                cue_candidates,
                temperature,
                policy_network,
            )

            if action.action_type == ActionType.STOP or state.budget <= 0:
                trajectory.steps.append(
                    RetrievalStep(
                        step_idx=t,
                        state={
                            "query": state.query,
                            "retrieved_count": len(state.retrieved_memories),
                            "budget": state.budget,
                        },
                        action={"action_type": ActionType.STOP.value},
                    )
                )
                break

            new_memories, retrieved_entry, cost = self._apply_action(
                action,
                state,
                client,
                primary_candidates,
                cue_candidates,
            )

            trajectory.steps.append(
                RetrievalStep(
                    step_idx=t,
                    state={
                        "query": state.query,
                        "retrieved_count": len(state.retrieved_memories),
                        "budget": state.budget,
                    },
                    action={
                        "action_type": action.action_type.value,
                        "target_memory_index": action.target_memory_index,
                        "score": action.score,
                    },
                )
            )
            state.retrieved_memories.extend(new_memories)

            if retrieved_entry is not None:
                if retrieved_entry.is_cue_index():
                    for primary_idx in retrieved_entry.get_linked_memories():
                        primary_entry = client._client._megamem.get(primary_idx)
                        if primary_entry:
                            trajectory.retrieved_memories.append({
                                "index": primary_entry.index,
                                "value": primary_entry.value,
                                "score": action.score,
                                "via_cue": retrieved_entry.index,
                            })
                else:
                    trajectory.retrieved_memories.append({
                        "index": retrieved_entry.index,
                        "value": retrieved_entry.value,
                        "score": action.score,
                    })

            state.budget -= cost
            total_cost += cost

        trajectory.cost = total_cost
        return trajectory

    def collect_trajectory_group(
        self,
        query: str,
        user_id: str,
        ground_truth: str = None,
        evidence: List[str] = None,
        G: int = 4,
        temperatures: List[float] = None,
    ) -> List[Trajectory]:
        if temperatures is None:
            temperatures = [0.0] + [0.3 + 0.2 * i for i in range(G - 1)]

        trajectories: List[Trajectory] = []
        for temp in temperatures[:G]:
            trajectories.append(
                self.collect_single_trajectory(
                    query=query,
                    user_id=user_id,
                    ground_truth=ground_truth,
                    evidence=evidence,
                    temperature=temp,
                    policy_network=None,
                )
            )

        return trajectories

    def _softmax(self, scores: List[float], temperature: float) -> List[float]:
        if not scores:
            return []
        scaled = [s / max(temperature, 1e-8) for s in scores]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        total = sum(exps)
        return [e / total for e in exps]
