from megamem.rl import Trajectory, TrajectoryScorer


def test_groundedness_prefers_evidence_identifier_recall() -> None:
    trajectory = Trajectory(
        query="What changed?",
        user_id="test",
        evidence=["doc:1", "doc:2"],
        retrieved_memories=[
            {"index": "doc:1", "value": "The policy changed."},
            {"index": "doc:3", "value": "Unrelated source."},
        ],
    )

    assert TrajectoryScorer().compute_groundedness(trajectory) == 0.5


def test_groundedness_falls_back_to_answer_token_recall() -> None:
    trajectory = Trajectory(
        query="When is the deadline?",
        user_id="test",
        ground_truth="Receipts are due within thirty days",
        retrieved_memories=[{"value": "All receipts are due within thirty days."}],
    )

    assert TrajectoryScorer().compute_groundedness(trajectory) == 1.0


def test_group_advantages_are_centered() -> None:
    scorer = TrajectoryScorer()
    trajectories = [
        Trajectory(query="q", user_id="u", ground_truth="alpha", retrieved_memories=[]),
        Trajectory(
            query="q",
            user_id="u",
            ground_truth="alpha",
            retrieved_memories=[{"value": "alpha"}],
        ),
    ]

    advantages = scorer.compute_group_advantages(trajectories)
    assert sum(advantages) == 0.0
    assert advantages[0] < advantages[1]
