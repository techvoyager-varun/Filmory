"""
Script to prepare a clean, reproducible temporal leave-last-out evaluation split.

Protocol (Kang & McAuley 2018 / He et al. 2017):
- For each user with history length >= 5:
  - Train sequence:  interactions[:-2]
  - Validation item: interactions[-2]
  - Test item:       interactions[-1] (held-out ground truth)
- 99 negative items sampled from items the user NEVER interacted with.

Ensures zero data leakage:
- Evaluator uses ONLY Train sequence to construct user states, profiles, and candidate pools.
- Saves manifest to backend/evaluation/split_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filmory.split")

BACKEND_DIR = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND_DIR / "ml"
EVAL_DIR = BACKEND_DIR / "evaluation"


def prepare_split(
    num_users: int = 1000,
    seed: int = 42,
    min_history: int = 5,
    num_negatives: int = 99,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    output_path = output_path or (EVAL_DIR / "split_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    logger.info("Loading ML artifacts from %s...", ML_DIR)
    with open(ML_DIR / "user_sequences.pkl", "rb") as f:
        user_sequences: Dict[int, List[int]] = pickle.load(f)

    with open(ML_DIR / "user_interacted.pkl", "rb") as f:
        user_interacted: Dict[int, Set[int]] = pickle.load(f)

    with open(ML_DIR / "model_config.json", "r") as f:
        model_config = json.load(f)

    total_items = model_config["num_items"]
    all_item_ids = list(range(total_items))

    # Eligible users: sequence length >= min_history
    eligible_user_ids = [
        u for u, seq in user_sequences.items() if len(seq) >= min_history
    ]
    logger.info("Found %d eligible users with >= %d interactions", len(eligible_user_ids), min_history)

    if num_users > 0 and len(eligible_user_ids) > num_users:
        selected_users = sorted(random.sample(eligible_user_ids, num_users))
    else:
        selected_users = sorted(eligible_user_ids)

    logger.info("Selected %d users for evaluation split (seed=%d)", len(selected_users), seed)

    user_splits = {}
    history_lengths = []

    for u in selected_users:
        seq = user_sequences[u]
        train_seq = seq[:-2]
        val_item = seq[-2]
        test_item = seq[-1]
        all_interacted = user_interacted.get(u, set(seq))

        # Sample 99 negatives that user never interacted with
        # Rejection sampling with exclusion set
        negatives = []
        while len(negatives) < num_negatives:
            candidate = random.choice(all_item_ids)
            if candidate not in all_interacted and candidate not in negatives:
                negatives.append(candidate)

        user_splits[str(u)] = {
            "user_idx": u,
            "train_history": train_seq,
            "val_item": val_item,
            "test_item": test_item,
            "negative_samples": negatives,
            "history_length": len(train_seq),
        }
        history_lengths.append(len(train_seq))

    manifest = {
        "metadata": {
            "protocol": "temporal_leave_last_out",
            "reference": "Kang & McAuley 2018 (SASRec), He et al. 2017 (NCF)",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "total_evaluated_users": len(selected_users),
            "num_negatives_per_user": num_negatives,
            "total_catalog_items": total_items,
            "history_stats": {
                "min": min(history_lengths) if history_lengths else 0,
                "mean": round(sum(history_lengths) / max(len(history_lengths), 1), 2),
                "max": max(history_lengths) if history_lengths else 0,
            },
        },
        "users": user_splits,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Split manifest successfully written to: %s", output_path)
    logger.info("Stats: %d users, avg train history = %.1f items", len(selected_users), manifest["metadata"]["history_stats"]["mean"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare temporal evaluation split")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users to evaluate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--min-history", type=int, default=5, help="Minimum interaction count per user (default: 5)")
    parser.add_argument("--num-neg", type=int, default=99, help="Number of negative samples per user (default: 99)")
    args = parser.parse_args()

    prepare_split(
        num_users=args.num_users,
        seed=args.seed,
        min_history=args.min_history,
        num_negatives=args.num_neg,
    )
