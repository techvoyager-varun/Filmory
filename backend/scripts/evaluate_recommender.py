"""
Scientific recommender evaluation for Filmory (diagnostic run).

Leakage status (see evaluation/provenance.json): the shipped checkpoints were
trained on ALL interactions, so learned-model numbers are OPTIMISTIC. The
evaluator removes every avoidable leak it controls:

  * profiles, genre vectors, popularity counts and DAMR states are rebuilt
    strictly from train-only histories (S[:-2]); shipped user_genre_matrix
    rows are never used as inputs;
  * DAMR state defaults to the clock-free positional estimator
    (state_variant="positional", freshness disabled — no timestamps exist);
    --state synthetic_time reproduces the old 90-day-spread simulation and is
    labelled as such;
  * the Bayesian quality prior is DISABLED in every variant
    (movies_metadata.csv carries no ratings/counts, so no training-only
    per-item quality signal exists);
  * the retrieval mixture is tuned on VALIDATION items of TUNE users, frozen,
    then applied to disjoint TEST users/items.

Track A (end-to-end): full-catalog Recall@100 for popularity / NCF baseline /
NCF hybrid / frozen val-tuned mixture, plus end-to-end HR@10/NDCG@10 for the
frozen mixture (+ DAMR) and popularity. A retrieval miss is a pipeline miss.
Track B (diagnostic, fixed 1+99 candidates, test users only): experts plus a
controlled DAMR ladder on IDENTICAL candidates —
Fixed Ensemble -> gate-only -> +momentum -> +agreement -> Full (+MMR).

Reads:  backend/evaluation/split_manifest.json
Writes: backend/evaluation/recommender_per_user.jsonl (tune rows + test rows)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filmory.eval_recommender")

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.model_service import model_service  # noqa: E402
from app.ml.damr import (  # noqa: E402
    _minmax,
    damr_rerank,
    estimate_user_state,
    estimate_user_state_from_positions,
    intra_list_diversity,
)

EVAL_DIR = BACKEND_DIR / "evaluation"

STATE_POSITIONAL = "positional"
STATE_SYNTHETIC_TIME = "synthetic_time"
QUALITY_PRIOR_STATUS = "disabled (no training-only per-item ratings available)"

TRACK_B_SYSTEMS = [
    "Popularity",
    "NCF Baseline",
    "NCF Hybrid",
    "Transformer",
    "Fixed Ensemble",
    "DAMR gate-only",
    "DAMR gate+momentum",
    "DAMR gate+momentum+agreement",
    "Full DAMR",
]
TRACK_B_SWITCHES = {
    # quality OFF in every rung (see QUALITY_PRIOR_STATUS); diversity via MMR only in Full.
    "Fixed Ensemble": {"use_adaptive": False, "use_momentum": False, "use_agreement": False, "use_quality": False, "use_diversity": False},
    "DAMR gate-only": {"use_adaptive": True, "use_momentum": False, "use_agreement": False, "use_quality": False, "use_diversity": False},
    "DAMR gate+momentum": {"use_adaptive": True, "use_momentum": True, "use_agreement": False, "use_quality": False, "use_diversity": False},
    "DAMR gate+momentum+agreement": {"use_adaptive": True, "use_momentum": True, "use_agreement": True, "use_quality": False, "use_diversity": False},
    "Full DAMR": {"use_adaptive": True, "use_momentum": True, "use_agreement": True, "use_quality": False, "use_diversity": True},
}

MIXTURE_GRID_STEP = 0.25


def mixture_grid() -> List[Dict[str, float]]:
    combos = []
    steps = [round(x * MIXTURE_GRID_STEP, 2) for x in range(int(1 / MIXTURE_GRID_STEP) + 1)]
    for a, b in itertools.product(steps, steps):
        if a + b <= 1.0:
            combos.append({"baseline": a, "hybrid": round(1.0 - a - b, 2), "popularity": b})
    return combos


def train_genre_vector(train_hist: Sequence[int]) -> torch.Tensor:
    """L2-normalised genre histogram rebuilt strictly from train history."""
    dev = model_service.device
    G = model_service.movie_genre_matrix.float()
    vec = G[torch.tensor([int(m) for m in train_hist], dtype=torch.long, device=dev)].sum(0)
    if float(vec.sum()) <= 0:
        return torch.zeros(G.shape[1], device=dev)
    return F.normalize(vec, dim=0)


def build_state(train_hist: Sequence[int], variant: str):
    if variant == STATE_SYNTHETIC_TIME:
        # SIMULATION (legacy): invented 90-day spacing; drift/freshness reflect
        # the spacing, not real activity. Labelled, never a temporal claim.
        now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        step = timedelta(days=90.0 / max(1, len(train_hist)))
        hist = [(int(m), now_dt - step * (len(train_hist) - i)) for i, m in enumerate(train_hist)]
        profile = estimate_user_state(
            history=hist, base_profile=None, now=now_dt,
            num_genres=model_service.config.get("num_genres", 20),
        )
    else:
        profile = estimate_user_state_from_positions(
            [int(m) for m in train_hist], base_profile=None,
            num_genres=model_service.config.get("num_genres", 20),
        )
    return profile


@torch.no_grad()
def full_catalog_scores(u_idx: int, u_gvec: torch.Tensor, dev) -> Dict[str, torch.Tensor]:
    """Train-masked full-catalog scores for the three retrieval sources.

    Transformer sequence handling (+1 shift) lives inside
    SequentialTransformer.score_candidates_with_sequence; callers pass RAW
    0-based model indices for candidates and 1-based sequence positions.
    """
    n_items = len(model_service.movie2idx)
    items = torch.arange(n_items, device=dev)
    out: Dict[str, torch.Tensor] = {}
    u = torch.full((n_items,), int(u_idx), dtype=torch.long, device=dev)
    out["baseline"] = model_service.ncf_baseline(u, items).float().cpu()
    ug = u_gvec.to(dev).unsqueeze(0).expand(n_items, -1)
    ig = model_service.movie_genre_matrix[items].float()
    out["hybrid"] = model_service.ncf_hybrid(u, items, ug, ig).float().cpu()
    return out


@torch.no_grad()
def transformer_scores(seq_1based: List[int], cand_idxs: torch.Tensor, dev) -> torch.Tensor:
    return model_service.sequential_transformer.score_candidates_with_sequence(
        sequence_item_indices=[int(x) for x in seq_1based],
        candidate_item_indices=cand_idxs,
        device=dev,
    ).float().cpu()


def genre_scores(cand_0based: torch.Tensor, u_gvec: torch.Tensor, dev) -> torch.Tensor:
    g = torch.mv(
        model_service.movie_genre_matrix[cand_0based.to(dev)].float(), u_gvec.to(dev)
    ).cpu()
    mx = float(g.max()) if g.numel() else 0.0
    return g / mx if mx > 0 else g


def rank_of(scores: torch.Tensor, target_pos: int) -> int:
    return int((scores > scores[target_pos]).sum().item()) + 1


def hr_ndcg(rank: int, k: int):
    hit = 1.0 if rank <= k else 0.0
    return hit, (1.0 / math.log2(rank + 1) if rank <= k else 0.0)


def run_evaluation(
    manifest_path: Path | None = None,
    tune_users: int = 100,
    sample_users: int = 200,
    top_k: int = 10,
    split_seed: int = 7,
    state_variant: str = STATE_POSITIONAL,
) -> Dict[str, Any]:
    manifest_path = manifest_path or (EVAL_DIR / "split_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("Loaded split manifest: %d users", len(manifest["users"]))

    model_service.load_all()
    dev = model_service.device
    n_items = len(model_service.movie2idx)

    # Train-only popularity (counts over train histories of manifest users).
    pop = torch.zeros(n_items)
    for u_info in manifest["users"].values():
        for m_idx in u_info["train_history"]:
            if 0 <= m_idx < n_items:
                pop[m_idx] += 1

    # Disjoint tune/test partition (seeded, recorded).
    all_users = sorted(manifest["users"].values(), key=lambda u: u["user_idx"])
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(all_users)).tolist()
    tune_pool = [all_users[i] for i in perm[: max(0, tune_users)]]
    test_pool = [all_users[i] for i in perm[max(0, tune_users):]]
    if 0 < sample_users < len(test_pool):
        test_users = [test_pool[i] for i in rng.permutation(len(test_pool))[:sample_users].tolist()]
    else:
        test_users = test_pool
    logger.info("Tune users: %d | Test users: %d | state=%s | quality=%s",
                len(tune_pool), len(test_users), state_variant, QUALITY_PRIOR_STATUS)

    # ---------------- Phase 1: tune retrieval mixture on VALIDATION ----------------
    combos = mixture_grid()
    val_recalls = {json.dumps(c, sort_keys=True): [] for c in combos}
    val_ranks = {json.dumps(c, sort_keys=True): [] for c in combos}
    tune_rows: List[Dict[str, Any]] = []
    for u_info in tune_pool:
        u_idx, train_hist, val_item = u_info["user_idx"], u_info["train_history"], u_info["val_item"]
        if not (0 <= val_item < n_items):
            continue
        u_gvec = train_genre_vector(train_hist)
        fc = full_catalog_scores(u_idx, u_gvec, dev)
        s_pop = pop.clone()
        train_mask = torch.zeros(n_items, dtype=torch.bool)
        train_mask[torch.tensor([m for m in train_hist if 0 <= m < n_items])] = True
        nb = _minmax(fc["baseline"])
        nh = _minmax(fc["hybrid"])
        denom = float(s_pop[~train_mask].max()) if (~train_mask).any() else 0.0
        npop = (s_pop / denom if denom > 0 else s_pop).clamp(0, 1)
        row: Dict[str, Any] = {"user_idx": u_idx, "split": "tune", "val_item": val_item,
                               "mixture_val": {}}
        for c in combos:
            key = json.dumps(c, sort_keys=True)
            mix = c["baseline"] * nb + c["hybrid"] * nh + c["popularity"] * npop
            mix[train_mask] = -1e9
            r = rank_of(mix, val_item)
            val_recalls[key].append(1.0 if r <= 100 else 0.0)
            val_ranks[key].append(r)
            row["mixture_val"][key] = {"rank": r, "recall100": r <= 100}
        tune_rows.append(row)

    scored = [(np.mean(val_recalls[k]), -np.mean(val_ranks[k]), k) for k in val_recalls]
    scored.sort(reverse=True)
    frozen_key = scored[0][2]
    frozen = json.loads(frozen_key)
    logger.info("Frozen mixture (val Recall@100=%.4f, mean val rank=%.1f): %s",
                scored[0][0], -scored[0][1], frozen)

    # ---------------- Phase 2: TEST users, both tracks ----------------
    track_a_rec: Dict[str, List[float]] = {k: [] for k in ("popularity", "baseline", "hybrid", "mixture")}
    e2e_hits, e2e_ndcgs, e2e_lat, e2e_recalls = [], [], [], []
    metrics = {s: {"hr": [], "ndcg": [], "ild": [], "items": set()} for s in TRACK_B_SYSTEMS}
    test_rows: List[Dict[str, Any]] = []

    for idx, u_info in enumerate(test_users):
        u_idx, train_hist = u_info["user_idx"], u_info["train_history"]
        pos_item, neg_samples = u_info["test_item"], u_info["negative_samples"]
        if not (0 <= pos_item < n_items):
            continue
        u_gvec = train_genre_vector(train_hist)
        seq_1based = [int(m) + 1 for m in train_hist]
        profile = build_state(train_hist, state_variant)

        # ---- Track A: full-catalog retrieval comparison ----
        fc = full_catalog_scores(u_idx, u_gvec, dev)
        train_mask = torch.zeros(n_items, dtype=torch.bool)
        train_mask[torch.tensor([m for m in train_hist if 0 <= m < n_items])] = True
        nb, nh = _minmax(fc["baseline"]), _minmax(fc["hybrid"])
        denom = float(pop[~train_mask].max()) if (~train_mask).any() else 0.0
        npop = (pop / denom if denom > 0 else pop).clamp(0, 1)
        cand_scores = {
            "popularity": npop.clone(),
            "baseline": nb.clone(),
            "hybrid": nh.clone(),
            "mixture": frozen["baseline"] * nb + frozen["hybrid"] * nh + frozen["popularity"] * npop,
        }
        retrieval_rec: Dict[str, Any] = {}
        for name, sc in cand_scores.items():
            sc = sc.clone()
            sc[train_mask] = -1e9
            r = rank_of(sc, pos_item)
            rec = 1.0 if r <= 100 else 0.0
            track_a_rec[name].append(rec)
            retrieval_rec[name] = {"rank": r, "recall100": bool(rec)}

        # ---- Track A end-to-end: frozen-mixture top-100 -> Full DAMR -> top-10 ----
        t0 = time.time()
        mix = cand_scores["mixture"].clone()
        mix[train_mask] = -1e9
        top100 = torch.topk(mix, 100).indices
        cands = top100.to(dev)
        e2e_res = damr_rerank(
            cand_idxs=cands,
            s_ncf=fc["hybrid"][top100].to(dev),
            s_tr=transformer_scores(seq_1based, cands, dev).to(dev),
            s_gen=genre_scores(cands, u_gvec, dev).to(dev),
            profile=profile, ratings=[None] * 100, counts=[None] * 100,
            top_k=top_k, **TRACK_B_SWITCHES["Full DAMR"],
        )
        e2e_top = [int(top100[e["pos"]]) for e in e2e_res]
        lat_ms = (time.time() - t0) * 1000
        if pos_item in e2e_top:
            e2e_h, e2e_nd = hr_ndcg(e2e_top.index(pos_item) + 1, top_k)
        else:
            e2e_h, e2e_nd = 0.0, 0.0
        e2e_hits.append(e2e_h)
        e2e_ndcgs.append(e2e_nd)
        e2e_lat.append(lat_ms)
        e2e_recalls.append(track_a_rec["mixture"][-1])

        # ---- Track B: fixed 1+99 candidates, controlled ladder ----
        eval_items = [pos_item] + [n for n in neg_samples if 0 <= n < n_items][:99]
        item_t = torch.tensor(eval_items, dtype=torch.long, device=dev)
        s_pop_b = pop[eval_items] / max(1.0, float(pop[eval_items].max()))
        s_base = model_service.ncf_baseline(
            torch.full((len(eval_items),), int(u_idx), dtype=torch.long, device=dev), item_t
        ).float().cpu()
        ug_exp = u_gvec.to(dev).unsqueeze(0).expand(len(eval_items), -1)
        s_hyb = model_service.ncf_hybrid(
            torch.full((len(eval_items),), int(u_idx), dtype=torch.long, device=dev),
            item_t, ug_exp, model_service.movie_genre_matrix[item_t].float(),
        ).float().cpu()
        s_tr = transformer_scores(seq_1based, item_t, dev)
        s_gen = genre_scores(item_t, u_gvec, dev)
        s_fixed = 0.55 * s_hyb + 0.25 * s_tr + 0.20 * s_gen

        def _order(t: torch.Tensor) -> List[int]:
            return torch.argsort(t, descending=True).cpu().tolist()

        rankings: Dict[str, List[int]] = {
            "Popularity": _order(s_pop_b),
            "NCF Baseline": _order(s_base),
            "NCF Hybrid": _order(s_hyb),
            "Transformer": _order(s_tr),
            "Fixed Ensemble": _order(s_fixed),
        }
        for v_name in ("DAMR gate-only", "DAMR gate+momentum",
                       "DAMR gate+momentum+agreement", "Full DAMR"):
            full = damr_rerank(
                cand_idxs=item_t, s_ncf=s_hyb, s_tr=s_tr, s_gen=s_gen,
                profile=profile, ratings=[None] * len(eval_items),
                counts=[None] * len(eval_items), top_k=len(eval_items),
                **TRACK_B_SWITCHES[v_name],
            )
            rankings[v_name] = [e["pos"] for e in full]

        sys_rec: Dict[str, Any] = {}
        for s_name in TRACK_B_SYSTEMS:
            ranking = rankings[s_name]
            r = ranking.index(0) + 1 if 0 in ranking else len(eval_items) + 1
            h, nd = hr_ndcg(r, top_k)
            top_model = [int(eval_items[p]) for p in ranking[:top_k]]
            ild = intra_list_diversity(top_model)
            assert 0.0 <= ild <= 2.0, f"ILD out of range: {ild}"
            metrics[s_name]["hr"].append(h)
            metrics[s_name]["ndcg"].append(nd)
            metrics[s_name]["ild"].append(ild)
            metrics[s_name]["items"].update(top_model)
            sys_rec[s_name] = {"hit": bool(h), "ndcg": round(nd, 4),
                               "rank": r if r <= len(eval_items) else None,
                               "ild": round(float(ild), 4), "top10": top_model}
        test_rows.append({
            "user_idx": u_idx, "split": "test", "train_history_len": len(train_hist),
            "state_variant": state_variant, "quality_prior": QUALITY_PRIOR_STATUS,
            "user_drift": round(float(profile.state.drift), 4),
            "user_maturity": round(float(profile.state.maturity), 4),
            "systems": sys_rec,
            "retrieval": retrieval_rec,
            "track_a_e2e": {"candidate_recall_100": bool(e2e_recalls[-1]),
                            "hit_10": bool(e2e_h), "ndcg_10": round(float(e2e_nd), 4),
                            "latency_ms": round(lat_ms, 2), "top10": e2e_top},
        })
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_users):
            logger.info("Evaluated %d/%d test users...", idx + 1, len(test_users))

    out_path = EVAL_DIR / "recommender_per_user.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in tune_rows:
            f.write(json.dumps({"split": "tune", **r}) + "\n")
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    summary: Dict[str, Any] = {
        "metadata": {
            "evaluated_test_users": len(test_rows), "tune_users": len(tune_rows),
            "catalog_items": n_items, "top_k": top_k,
            "split_seed": split_seed, "state_variant": state_variant,
            "quality_prior": QUALITY_PRIOR_STATUS,
            "frozen_mixture": frozen,
            "val_mixture_recall100": round(float(scored[0][0]), 4),
            "provenance": "see evaluation/provenance.json (DIAGNOSTIC: checkpoints trained on all interactions)",
        },
        "track_a_retrieval_recall100": {
            k: round(float(np.mean(v)), 4) for k, v in track_a_rec.items()},
        "track_a_end_to_end": {
            "retriever": "frozen val-tuned mixture + Full DAMR (no quality)",
            "candidate_recall_100": round(float(np.mean(e2e_recalls)), 4),
            "hr_10": round(float(np.mean(e2e_hits)), 4),
            "ndcg_10": round(float(np.mean(e2e_ndcgs)), 4),
            "latency_p50_ms": round(float(np.percentile(e2e_lat, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(e2e_lat, 95)), 2),
        },
        "track_b_fixed_candidate_ablation": {},
    }
    for s in TRACK_B_SYSTEMS:
        m = metrics[s]
        summary["track_b_fixed_candidate_ablation"][s] = {
            "HR@10": round(float(np.mean(m["hr"])), 4),
            "NDCG@10": round(float(np.mean(m["ndcg"])), 4),
            "ILD@10": round(float(np.mean(m["ild"])), 4),
            "Catalog_Coverage_pct": round(len(m["items"]) / n_items * 100, 2),
        }
    logger.info("Wrote %s", out_path)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Filmory recommender (diagnostic)")
    p.add_argument("--tune-users", type=int, default=100)
    p.add_argument("--sample-users", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--state", choices=[STATE_POSITIONAL, STATE_SYNTHETIC_TIME],
                   default=STATE_POSITIONAL)
    args = p.parse_args()
    print(json.dumps(run_evaluation(tune_users=args.tune_users, sample_users=args.sample_users,
                                    top_k=args.top_k, split_seed=args.split_seed,
                                    state_variant=args.state), indent=2))
