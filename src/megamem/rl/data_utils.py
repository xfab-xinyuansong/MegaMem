"""Helpers for loading/splitting LoCoMo data into QA pairs and persisting
trajectories collected during RL rollouts."""

import json
from typing import Dict, List, Tuple
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class QAPair:
    """One question-answer record paired with its source conversation context."""
    query: str
    answer: str
    category: str
    evidence: List[str]
    conv_idx: int
    speaker_a: str
    speaker_b: str
    user_id: str


def load_and_split_locomo(
        data_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load LoCoMo and partition it conversation-wise into train / val / test.

    Returns:
        Tuple of (train_data, val_data, test_data) — each entry is a list of
        per-conversation dicts.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    random.seed(seed)

    with open(data_path, 'r') as fh:
        raw = json.load(fh)

    # The dataset has shipped in both dict-keyed and list form; normalise.
    if isinstance(raw, dict):
        raw = list(raw.values())

    n_total = len(raw)
    print('===' * 20)
    print(f"Loaded {n_total} conversations from {data_path}")
    print('===' * 20)

    indices = list(range(n_total))
    random.shuffle(indices)

    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_data = [raw[i] for i in train_idx]
    val_data = [raw[i] for i in val_idx]
    test_data = [raw[i] for i in test_idx]

    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def extract_qa_pairs(conversation_data: List[Dict]) -> List[QAPair]:
    """
    Flatten a list of conversation dicts into QA pairs ready for trajectory
    collection.

    Returns:
        List of fully-populated :class:`QAPair` objects.
    """
    qa_pairs: List[QAPair] = []
    for conv_idx, item in enumerate(conversation_data):
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", f"speaker_a_{conv_idx}")
        speaker_b = conversation.get("speaker_b", f"speaker_b_{conv_idx}")

        user_id = f"{speaker_a}_{speaker_b}_{conv_idx}"

        for qa in item.get("qa", []):
            # Some adversarial-only items (category 5) only carry an
            # ``adversarial_answer``; fall back gracefully.
            answer = qa.get("answer") or qa.get("adversarial_answer", "")

            if not answer:
                continue

            qa_pairs.append(
                QAPair(
                    query=qa["question"],
                    # A handful of answers are integers — cast to string.
                    answer=str(answer),
                    category=str(qa.get("category", "unknown")),
                    evidence=qa.get("evidence", []),
                    conv_idx=conv_idx,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    user_id=user_id,
                )
            )
    return qa_pairs


def save_trajectories(trajectories: List[Dict], output_path: str):
    """Write ``trajectories`` to ``output_path`` as pretty JSON."""
    with open(output_path, 'w') as fh:
        json.dump(trajectories, fh, indent=2)
    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def load_trajectories(input_path: str) -> List[Dict]:
    """Read previously saved trajectories from a JSON file."""
    with open(input_path, 'r') as fh:
        return json.load(fh)


if __name__ == "__main__":
    data_path = str(Path(__file__).parent.parent / "data" / "locomo10.json")
    train_data, val_data, test_data = load_and_split_locomo(data_path)

    train_qa = extract_qa_pairs(train_data)
    val_qa = extract_qa_pairs(val_data)
    test_qa = extract_qa_pairs(test_data)

    print(f"Train QA pairs: {len(train_qa)}")
    print(f"Validation QA pairs: {len(val_qa)}")
    print(f"Test QA pairs: {len(test_qa)}")

    if train_qa:
        sample = train_qa[0]
        print(f"\nSample QA pair:")
        print(f"  user_id: {sample.user_id}")
        print(f"  query: {sample.query}")
        print(f"  answer: {sample.answer}")
        print(f"  evidence: {sample.evidence}")
