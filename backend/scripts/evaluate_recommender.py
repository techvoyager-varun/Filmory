"""
Scientific Recommender Evaluation Script for Filmory.

Implements two distinct evaluation tracks:
Track A: End-to-End Candidate Retrieval + Ranking (Measures Candidate Recall@100, HR@10, NDCG@10 on full catalog retrieval).
Track B: Fixed-Candidate Diagnostic Ablation (1 held-out positive vs 99 unobserved sampled negatives).

Evaluated Systems:
- Popularity Baseline
- NCF Baseline (Collaborative Filtering)
- NCF Hybrid (+ Genre projections)
- Sequential Transformer (SASRec-style sequential expert)
- Fixed Ensemble (0.55 NCF + 0.25 TR + 0.20 Genre)
- DAMR −adaptive (Ablation without dynamic gate)
- DAMR −momentum (Ablation without taste momentum)
- Full DAMR (Dynamic Gate + Taste Momentum + Quality Prior + MMR Diversity)

Reads from: backend/evaluation/split_manifest.json
Writes to:   backend/evaluation/recommender_per_user.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filmory.eval_recommender")

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.ml.model_service import model_service
from app.ml.damr import (
    TasteProfile,
    UserState,
    damr_rerank,
    estimate_user_state,
    switches_for_variant,
    intra_list_diversity,
)

EVAL_DIR = BACKEND_DIR / "evaluation"


def score_popularity(item_idxs: torch.Tensor, pop_tensor: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    p = pop_tensor[item_idxs.to(dev)]
    mx = p.max()
    return p / mx if mx > 0 else p


def score_ncf_baseline(u_idx: int, item_idxs: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    model = model_service.ncf_baseline
    u_t = torch.tensor([u_idx] * len(item_idxs), dtype=torch.long, device=dev)
    with torch.no_grad():
        preds = model(u_t, item_idxs.to(dev))
    return preds.squeeze(-1)


def score_ncf_hybrid(u_idx: int, item_idxs: torch.Tensor, u_gvec: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    model = model_service.ncf_hybrid
    u_t = torch.tensor([u_idx] * len(item_idxs), dtype=torch.long, device=dev)
    item_g = model_service.movie_genre_matrix[item_idxs.to(dev)].float()
    u_g_expanded = u_gvec.unsqueeze(0).expand(len(item_idxs), -1)
    with torch.no_grad():
        preds = model(u_t, item_idxs.to(dev), u_g_expanded, item_g)
    return preds.squeeze(-1)


def score_transformer(hist_item_idxs: Sequence[int], item_idxs: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    model = model_service.sequential_transformer
    with torch.no_grad():
        preds = model.score_candidates_with_sequence(
            sequence_item_indices=list(hist_item_idxs),
            candidate_item_indices=item_idxs,
            device=dev,
        )
    return preds


def score_genre(item_idxs: torch.Tensor, u_gvec: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    g = torch.mv(model_service.movie_genre_matrix[item_idxs.to(dev)].float(), u_gvec.to(dev))
    mx = g.max()
    return g / mx if mx > 0 else g


def run_evaluation(
    manifest_path: Path | None = None,
    sample_users: int = 0,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    manifest_path = manifest_path or (EVAL_DIR / "split_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_evaluation_split.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info("Loaded split manifest: %d users, seed=%s", len(manifest["users"]), manifest["metadata"]["random_seed"])

    logger.info("Loading PyTorch models & artifacts into model_service...")
    model_service.load_all()
    dev = model_service.device

    movie2idx = model_service.movie2idx
    user2idx = model_service.user2idx
    n_items = len(movie2idx)

    # Popularity vector
    pop = torch.zeros(n_items, device=dev)
    for u_info in manifest["users"].values():
        for m_idx in u_info["train_history"]:
            if m_idx < n_items:
                pop[m_idx] += 1

    users_data = list(manifest["users"].values())
    if sample_users > 0 and sample_users < len(users_data):
        torch.manual_seed(seed)
        idx_perm = torch.randperm(len(users_data))[:sample_users].tolist()
        users_data = [users_data[i] for i in idx_perm]

    logger.info("Starting evaluation across %d users (K=%d)...", len(users_data), top_k)

    SYSTEMS = [
        "Popularity",
        "NCF Baseline",
        "NCF Hybrid",
        "Transformer",
        "Fixed Ensemble",
        "DAMR −adaptive",
        "DAMR −momentum",
        "Full DAMR",
    ]

    metrics = {
        name: {
            "hr": [],
            "ndcg": [],
            "mrr": [],
            "ild": [],
            "top_items": set(),
        }
        for name in SYSTEMS
    }

    # Track A: End-to-End metrics (real candidate retrieval over entire catalog)
    track_a_metrics = {
        "candidate_recall_100": [],
        "hr_10": [],
        "ndcg_10": [],
        "latencies_ms": [],
    }

    per_user_output_path = EVAL_DIR / "recommender_per_user.jsonl"
    jsonl_file = open(per_user_output_path, "w", encoding="utf-8")

    t_start = time.time()

    for idx, u_info in enumerate(users_data):
        u_idx = u_info["user_idx"]
        train_hist = u_info["train_history"]
        pos_item = u_info["test_item"]
        neg_samples = u_info["negative_samples"]

        # Ensure valid mapping
        if pos_item >= n_items or u_idx >= len(user2idx):
            continue

        # -------------------------------------------------------------------
        # Track B: Fixed Candidates Diagnostic (1 Pos + 99 Negs)
        # -------------------------------------------------------------------
        eval_items = [pos_item] + neg_samples
        item_idxs = torch.tensor(eval_items, dtype=torch.long, device=dev)
        u_gvec = F.normalize(model_service.user_genre_matrix[u_idx].float(), dim=0).to(dev)

        # 1. Base experts
        s_pop = score_popularity(item_idxs, pop)
        s_ncf_base = score_ncf_baseline(u_idx, item_idxs)
        s_ncf_hyb = score_ncf_hybrid(u_idx, item_idxs, u_gvec)
        s_tr = score_transformer(train_hist, item_idxs)
        s_gen = score_genre(item_idxs, u_gvec)

        # 2. Fixed Ensemble (0.55 NCF + 0.25 TR + 0.20 Genre)
        s_fixed = 0.55 * s_ncf_hyb + 0.25 * s_tr + 0.20 * s_gen

        # User profile constructed strictly from train history
        now_dt = datetime.now(timezone.utc)
        step = 90.0 / max(1, len(train_hist))
        hist_tuples = [
            (m, now_dt - (len(train_hist) - i) * (now_dt - now_dt))  # dummy timedelta simulation
            for i, m in enumerate(train_hist)
        ]
        user_state, profile = estimate_user_state(
            history=hist_tuples,
            movie_genres=model_service.movie_genre_matrix,
            num_genres=model_service.config.get("num_genres", 20),
            total_history_count=len(train_hist),
            now=now_dt,
        )

        ratings_list = [model_service.idx2movie.get(m, {}).get("rating") for m in eval_items] if hasattr(model_service, "idx2movie") and isinstance(model_service.idx2movie, dict) else [None] * len(eval_items)
        counts_list = [model_service.idx2movie.get(m, {}).get("rating_count") for m in eval_items] if hasattr(model_service, "idx2movie") and isinstance(model_service.idx2movie, dict) else [None] * len(eval_items)

        # System predictions dictionary: {sys_name: ranked_indices_in_eval_items}
        system_rankings: Dict[str, List[int]] = {}

        def _rank_by_score(s_tensor: torch.Tensor) -> List[int]:
            return torch.argsort(s_tensor, descending=True).cpu().tolist()

        system_rankings["Popularity"] = _rank_by_score(s_pop)
        system_rankings["NCF Baseline"] = _rank_by_score(s_ncf_base)
        system_rankings["NCF Hybrid"] = _rank_by_score(s_ncf_hyb)
        system_rankings["Transformer"] = _rank_by_score(s_tr)
        system_rankings["Fixed Ensemble"] = _rank_by_score(s_fixed)

        # DAMR variants
        for v_name in ["DAMR −adaptive", "DAMR −momentum", "Full DAMR"]:
            switches = {
                "DAMR −adaptive": {"use_adaptive": False, "use_momentum": True, "use_diversity": True},
                "DAMR −momentum": {"use_adaptive": True, "use_momentum": False, "use_diversity": True},
                "Full DAMR": {"use_adaptive": True, "use_momentum": True, "use_diversity": True},
            }[v_name]

            res_order = damr_rerank(
                cand_idxs=item_idxs,
                s_ncf=s_ncf_hyb,
                s_tr=s_tr,
                s_gen=s_gen,
                profile=profile,
                ratings=ratings_list,
                counts=counts_list,
                top_k=top_k,
                **switches,
            )
            # res_order contains positions in eval_items
            system_rankings[v_name] = [item["pos"] for item in res_order]

        # Calculate metrics for each system on pos_item (index 0 in eval_items)
        per_user_record = {
            "user_idx": u_idx,
            "train_history_len": len(train_hist),
            "user_drift": round(float(user_state.drift), 4),
            "user_maturity": round(float(user_state.maturity), 4),
            "systems": {},
        }

        for s_name in SYSTEMS:
            ranking = system_rankings[s_name]
            # Find 1-based rank of positive item (index 0)
            if 0 in ranking[:top_k]:
                rank_1based = ranking[:top_k].index(0) + 1
                hr = 1.0
                ndcg = 1.0 / math.log2(rank_1based + 1)
                mrr = 1.0 / rank_1based
            else:
                hr = 0.0
                ndcg = 0.0
                mrr = 0.0

            # Intra-list diversity
            top_cand_tensor = item_idxs[ranking[:top_k]]
            ild = intra_list_diversity(top_cand_tensor, model_service.movie_genre_matrix)

            metrics[s_name]["hr"].append(hr)
            metrics[s_name]["ndcg"].append(ndcg)
            metrics[s_name]["mrr"].append(mrr)
            metrics[s_name]["ild"].append(ild)
            for r_pos in ranking[:top_k]:
                metrics[s_name]["top_items"].add(eval_items[r_pos])

            per_user_record["systems"][s_name] = {
                "hit": hr == 1.0,
                "ndcg": round(ndcg, 4),
                "rank": ranking.index(0) + 1 if 0 in ranking else None,
            }

        # -------------------------------------------------------------------
        # Track A: End-to-End Evaluation (Full Catalog Candidate Gen + DAMR)
        # -------------------------------------------------------------------
        t_req_start = time.time()
        # Candidate pool generation using model's top items excluding train history
        seen_train = set(train_hist)
        with torch.no_grad():
            # Combine NCF + TR over entire catalog for candidates
            u_tens = torch.tensor([u_idx], device=dev)
            cand_scores = model_service.ncf_hybrid(
                u_tens.repeat(n_items),
                torch.arange(n_items, device=dev),
                u_gvec.unsqueeze(0).repeat(n_items, 1),
                model_service.movie_genre_matrix.float().to(dev),
            ).squeeze(-1)
            # Mask out training history
            for seen_m in seen_train:
                if seen_m < n_items:
                    cand_scores[seen_m] = -1e9

            top100_cands = torch.topk(cand_scores, 100).indices.cpu().tolist()

        # Check Candidate Recall@100
        cand_recall_100 = 1.0 if pos_item in top100_cands else 0.0
        track_a_metrics["candidate_recall_100"].append(cand_recall_100)

        # Full DAMR reranking over top 100 retrieved candidates
        c_tens = torch.tensor(top100_cands, dtype=torch.long, device=dev)
        e2e_res = damr_rerank(
            cand_idxs=c_tens,
            s_ncf=cand_scores[c_tens],
            s_tr=score_transformer(train_hist, c_tens),
            s_gen=score_genre(c_tens, u_gvec),
            profile=profile,
            ratings=[None] * len(top100_cands),
            counts=[None] * len(top100_cands),
            top_k=top_k,
            use_adaptive=True,
            use_momentum=True,
            use_diversity=True,
        )
        e2e_top10 = [top100_cands[item["pos"]] for item in e2e_res]

        if pos_item in e2e_top10:
            e2e_rank = e2e_top10.index(pos_item) + 1
            e2e_hr = 1.0
            e2e_ndcg = 1.0 / math.log2(e2e_rank + 1)
        else:
            e2e_hr = 0.0
            e2e_ndcg = 0.0

        track_a_metrics["hr_10"].append(e2e_hr)
        track_a_metrics["ndcg_10"].append(e2e_ndcg)
        track_a_metrics["latencies_ms"].append((time.time() - t_req_start) * 1000)

        per_user_record["track_a_e2e"] = {
            "candidate_recall_100": cand_recall_100,
            "hit_10": e2e_hr == 1.0,
            "ndcg_10": round(e2e_ndcg, 4),
        }

        jsonl_file.write(json.dumps(per_user_record) + "\n")

        if (idx + 1) % 50 == 0 or (idx + 1) == len(users_data):
            logger.info("Evaluated %d/%d users... (current E2E HR@10=%.3f)", idx + 1, len(users_data), np.mean(track_a_metrics["hr_10"]))

    jsonl_file.close()
    elapsed_total = time.time() - t_start

    # Compile results
    results_summary: Dict[str, Any] = {
        "metadata": {
            "evaluated_users": len(users_data),
            "catalog_items": n_items,
            "top_k": top_k,
            "elapsed_seconds": round(elapsed_total, 2),
        },
        "track_a_end_to_end": {
            "candidate_recall_100": round(float(np.mean(track_a_metrics["candidate_recall_100"])), 4),
            "hr_10": round(float(np.mean(track_a_metrics["hr_10"])), 4),
            "ndcg_10": round(float(np.mean(track_a_metrics["ndcg_10"])), 4),
            "latency_p50_ms": round(float(np.percentile(track_a_metrics["latencies_ms"], 50)), 2),
            "latency_p95_ms": round(float(np.percentile(track_a_metrics["latencies_ms"], 95)), 2),
        },
        "track_b_fixed_candidate_ablation": {},
    }

    for s_name in SYSTEMS:
        hr_mean = float(np.mean(metrics[s_name]["hr"]))
        ndcg_mean = float(np.mean(metrics[s_name]["ndcg"]))
        mrr_mean = float(np.mean(metrics[s_name]["mrr"]))
        ild_mean = float(np.mean(metrics[s_name]["ild"]))
        coverage_pct = round((len(metrics[s_name]["top_items"]) / n_items) * 100, 2)

        results_summary["track_b_fixed_candidate_ablation"][s_name] = {
            "HR@10": round(hr_mean, 4),
            "NDCG@10": round(ndcg_mean, 4),
            "MRR": round(mrr_mean, 4),
            "ILD@10": round(ild_mean, 4),
            "Catalog_Coverage_pct": coverage_pct,
        }

    logger.info("Recommender evaluation complete. Written to %s", per_user_output_path)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Filmory Recommender")
    parser.add_argument("--sample-users", type=int, default=200, help="Number of users to evaluate (0 for all in split)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    res = run_evaluation(sample_users=args.sample_users, top_k=args.top_k, seed=args.seed)
    print(json.dumps(res, indent=2))
