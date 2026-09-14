"""
Record training-artifact provenance for the Filmory evaluation suite.

Hashes every checkpoint / mapping / matrix the evaluation depends on and
records the (uncomfortable but load-bearing) leakage fact established by audit:

  * No training script or training split ships with this repository.
  * docs/DAMR.md already states the shipped checkpoints were trained on ALL
    interactions, including any held-out target of a later split.
  * user_sequences.pkl / user_interacted.pkl keys AND values are model-index
    space; user_genre_matrix rows are exactly the full-sequence genre
    histograms (cosine 1.0 to full-history histograms) — i.e. training-derived
    artifacts that contain the held-out items.
  * No interaction timestamps ship with the ML artifacts (order only).

Consequence: any holdout number computed with these checkpoints is
DIAGNOSTIC/OPTIMISTIC for the learned models, never a clean generalization
claim. A strictly clean number requires retraining on a documented clean
split (training code + timestamps are not available in this repo).

Writes: backend/evaluation/provenance.json
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND_DIR / "ml"
EVAL_DIR = BACKEND_DIR / "evaluation"

TRACKED_FILES = [
    "ml/ncf_baseline.pth",
    "ml/ncf_hybrid.pth",
    "ml/sequential_transformer.pth",
    "ml/user_sequences.pkl",
    "ml/user_interacted.pkl",
    "ml/movie_genre_matrix.pt",
    "ml/user_genre_matrix.pt",
    "ml/model_config.json",
    "ml/movie2idx.pkl",
    "ml/user2idx.pkl",
    "ml/idx2movie.pkl",
    "ml/idx2user.pkl",
    "ml/genre2idx.pkl",
    "ml/idx2genre.pkl",
    "ml/movies_metadata.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = {}
    for rel in TRACKED_FILES:
        p = BACKEND_DIR / rel
        if p.exists():
            files[rel] = {
                "sha256": sha256_file(p),
                "bytes": os.path.getsize(p),
            }
        else:
            files[rel] = {"missing": True}

    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "training_code_available": False,
        "training_split_available": False,
        "interaction_timestamps_available": False,
        "leakage_status": (
            "DIAGNOSTIC — shipped checkpoints were trained on ALL interactions "
            "(docs/DAMR.md section 6). Creating a holdout split after training "
            "does not remove held-out information from weights, user_genre_matrix "
            "rows (exact full-sequence histograms), or user_interacted sets. "
            "Learned-model numbers are OPTIMISTIC. Clean generalization claims "
            "require retraining on a documented clean split."
        ),
        "mitigations_applied_by_evaluator": [
            "user profiles, genre vectors, popularity counts and DAMR states are "
            "rebuilt strictly from train-only histories (S[:-2]); shipped "
            "user_genre_matrix rows are NOT used as model inputs",
            "DAMR state uses the clock-free positional estimator with "
            "freshness disabled (no timestamps exist); labelled 'positional'",
            "Bayesian quality prior disabled (movies_metadata.csv carries no "
            "ratings/counts; no training-only per-item quality signal exists)",
            "retrieval mixture tuned on validation items, frozen, then applied "
            "to disjoint test users/items",
        ],
        "files": files,
    }
    out = EVAL_DIR / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out} ({len(files)} artifacts hashed)")


if __name__ == "__main__":
    main()
