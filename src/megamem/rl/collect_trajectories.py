"""Driver that walks every QA pair in the LoCoMo split and gathers a group
of retrieval trajectories per query for downstream GRPO training."""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_utils import load_and_split_locomo, extract_qa_pairs, save_trajectories
from trajectory_utils import TrajectoryCollector, Trajectory


def collect_all_trajectories(
    cfg: DictConfig,
    qa_pairs,
    G: int = 4,
    output_path: str = None,
):
    """
    Collect a group of ``G`` trajectories for every QA pair.

    Args:
        cfg: Hydra config
        qa_pairs: List of QAPair objects
        G: Number of trajectories per query
        output_path: Where to save trajectories
    """
    rl_cfg = cfg.get("rl", {})
    collector = TrajectoryCollector(
        cfg=cfg,
        top_k=cfg.memory.get("top_k", 20),
        budget=rl_cfg.get("budget", 10.0),
        max_steps=rl_cfg.get("max_steps", 15),
    )

    aggregated: list = []

    for qa in tqdm(qa_pairs, desc="Collecting trajectories"):
        try:
            group = collector.collect_trajectory_group(
                query=qa.query,
                user_id=qa.user_id,
                ground_truth=qa.answer,
                evidence=qa.evidence,
                G=G,
            )

            for traj in group:
                aggregated.append(traj.to_dict())

        except Exception as e:
            print(f"Error collecting trajectory for query '{qa.query}': {e}")
            continue

    if output_path:
        save_trajectories(aggregated, output_path)

    return aggregated


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point for trajectory collection."""

    data_path = str(Path(__file__).parent.parent / "data" / "locomo10.json")
    train_data, val_data, _test_data = load_and_split_locomo(data_path)

    use_combined_user = cfg.eval.get("use_combined_user", True)
    train_qa = extract_qa_pairs(train_data, use_combined_user=use_combined_user)
    val_qa = extract_qa_pairs(val_data, use_combined_user=use_combined_user)

    print(f"Train QA pairs: {len(train_qa)}")
    print(f"Validation QA pairs: {len(val_qa)}")

    output_dir = Path(__file__).parent / "trajectories"
    output_dir.mkdir(exist_ok=True)

    G = cfg.get("rl", {}).get("G", 4)  # trajectories per query
    print(f"\n=== Collecting {G} trajectories per query ===")

    train_trajectories = collect_all_trajectories(
        cfg=cfg,
        qa_pairs=train_qa,
        G=G,
        output_path=str(output_dir / "train_trajectories.json"),
    )

    val_trajectories = collect_all_trajectories(
        cfg=cfg,
        qa_pairs=val_qa,
        G=G,
        output_path=str(output_dir / "val_trajectories.json"),
    )

    print(f"\nCollected {len(train_trajectories)} training trajectories")
    print(f"Collected {len(val_trajectories)} validation trajectories")


if __name__ == "__main__":
    main()
