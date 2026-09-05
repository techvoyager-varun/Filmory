"""
Offline evaluation for Filmory's recommendation models.

Protocol: temporal leave-one-out (the standard NCF / SASRec protocol —
He et al. 2017, Kang & McAuley 2018). For every user:
  - hold out their LAST interaction as the test positive,
  - sample 99 movies the user never interacted with as negatives,
  - score the 1 positive + 99 negatives, rank them,
  - report where the positive landed.

Implicit-feedback recommenders are NOT measured with plain classification
accuracy (99%+ of user-item pairs are negatives, so a model that always
predicts "no" is trivially "99% accurate"). The standard metrics are:

  HR@K      is the held-out movie inside the top-K?
  NDCG@K    same, but rewards ranking it higher (log discount)
  MRR       1 / rank of the held-out movie
  AUC       P(positive scored above a random negative)

Evaluated systems:
  Popularity        non-personalised baseline
  NCF Baseline      pure collaborative filtering
  NCF Hybrid        + genre projections (Stage 1 expert)
  Transformer       sequential expert (Stage 2 expert)
  Genre             genre-affinity expert (Stage 3 expert)
  Static Ensemble   0.55 NCF + 0.25 TR + 0.20 Genre      (legacy Stage 1-3)
  Static + MMR      ensemble + quality prior + MMR       (engineering baseline)
  DAMR (ours)       Stage 4: drift-aware adaptive fusion + momentum
                    + agreement + quality + diversity

Beyond-accuracy metrics for the list-producing systems:
  ILD@10            intra-list diversity (avg pairwise 1 - similarity)
  Coverage          % of the catalogue that appears in anyone's top-10

Usage:
  cd backend && python scripts/evaluate.py [--num-users 2000] [--num-neg 99]
                                           [--k 10] [--seed 42] [--ablation]

Outputs:
  ml/metrics.json         the numbers (served by GET /api/metrics)
  ml/metrics_by_user.csv  per-user state + outcomes (segment analysis)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- make `app` importable when run as a script -----------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.ml.model_service import model_service  # noqa: E402
from app.ml.damr import (  # noqa: E402
    TasteProfile,
    UserState,
    damr_rerank,
    estimate_user_state,
    switches_for_variant,
    item_pair_similarity,
    intra_list_diversity,
)

K = 10
NUM_NEG = 99
SYNTH_SPREAD_DAYS = 90.0  # matches build_taste_profile's fallback spread


# ---------------------------------------------------------------------------
# Metrics (implemented from scratch — no sklearn dependency)
# ---------------------------------------------------------------------------
def auc_from_scores(y: List[int], p: Sequence[float]) -> float:
    """AUC via the Mann-Whitney U statistic with average ranks for ties."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos_ranks = sum(r for r, yy in zip(ranks, y) if yy == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def hr_ndcg_mrr(rank: int, k: int) -> Tuple[float, float, float]:
    """rank is 1-based; rank > k means the positive was outside the top-k."""
    hit = 1.0 if rank <= k else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    mrr = 1.0 / rank
    return hit, ndcg, mrr


# ---------------------------------------------------------------------------
# Scoring helpers for each expert
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_popularity(item_idxs: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
    return pop[item_idxs].to(pop.device)


@torch.no_grad()
def score_ncf_baseline(u_idx: int, item_idxs: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    u = torch.full((len(item_idxs),), u_idx, dtype=torch.long, device=dev)
    return model_service.ncf_baseline(u, item_idxs.to(dev)).float()


@torch.no_grad()
def score_ncf_hybrid(u_idx: int, item_idxs: torch.Tensor, u_gvec: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    items = item_idxs.to(dev)
    k = len(items)
    u = torch.full((k,), u_idx, dtype=torch.long, device=dev)
    ug = u_gvec.to(dev).unsqueeze(0).expand(k, -1)
    ig = model_service.movie_genre_matrix[items].float()
    return model_service.ncf_hybrid(u, items, ug, ig).float()


@torch.no_grad()
def score_transformer(history_movie_ids: Sequence[int], item_idxs: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    seq = [model_service.movie2idx[m] + 1 for m in history_movie_ids if m in model_service.movie2idx]
    return model_service.sequential_transformer.score_candidates_with_sequence(
        sequence_item_indices=seq,
        candidate_item_indices=item_idxs.to(dev),
        device=dev,
    ).float()


@torch.no_grad()
def score_genre(item_idxs: torch.Tensor, u_gvec: torch.Tensor) -> torch.Tensor:
    dev = model_service.device
    g = torch.mv(model_service.movie_genre_matrix[item_idxs.to(dev)].float(), u_gvec.to(dev))
    mx = g.max()
    return g / mx if mx > 0 else g


def synth_history(seq: Sequence[int], spread_days: float = SYNTH_SPREAD_DAYS) -> List[Tuple[int, datetime]]:
    """
    Convert a chronological movieId sequence into timestamped (model_idx, ts)
    history the same way the serving fallback does (latest item = now).
    """
    mapped = [model_service.movie2idx[m] for m in seq if m in model_service.movie2idx]
    if not mapped:
        return []
    now = datetime.utcnow()
    step = timedelta(days=spread_days / max(1, len(mapped)))
    return [(idx, now - step * (len(mapped) - k)) for k, idx in enumerate(mapped)]


def rerank_variant(
    variant: str,
    cand_idxs: torch.Tensor,
    s_ncf: torch.Tensor,
    s_tr: torch.Tensor,
    s_gen: torch.Tensor,
    profile: TasteProfile,
    ratings: List[Optional[float]],
    counts: List[Optional[int]],
    top_k: int,
) -> Tuple[List[int], List[float]]:
    """
    Run a Stage-4 variant over the candidate pool in a single greedy pass.
    Returns (greedy_selection_order, relevance_scores_in_candidate_order).
    order[0] is the best item; relevance is the pre-diversity ranking signal
    (used for the AUC computation).
    """
    full_order = damr_rerank(
        cand_idxs=cand_idxs, s_ncf=s_ncf, s_tr=s_tr, s_gen=s_gen,
        profile=profile, ratings=ratings, counts=counts,
        top_k=len(cand_idxs),  # greedy order over the whole pool
        **switches_for_variant(variant),
    )
    order = [e["pos"] for e in full_order]
    rel_map = {e["pos"]: e["score"] for e in full_order}
    scores = [rel_map[i] for i in range(len(cand_idxs))]
    return order, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global K, NUM_NEG
    parser = argparse.ArgumentParser(description="Filmory offline evaluation")
    parser.add_argument("--num-users", type=int, default=2000, help="users to evaluate (0 = all)")
    parser.add_argument("--num-neg", type=int, default=99)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", action="store_true", help="also evaluate DAMR ablations")
    parser.add_argument("--min-history", type=int, default=5, help="min sequence length per user")
    parser.add_argument("--out", default=None, help="output JSON path (default ml/metrics.json)")
    args = parser.parse_args()

    K = args.k
    NUM_NEG = args.num_neg
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading models & artifacts ...")
    model_service.load_all()
    dev = model_service.device

    movie2idx: Dict[int, int] = model_service.movie2idx
    user2idx: Dict[int, int] = model_service.user2idx
    user_sequences: Dict[int, List[int]] = model_service.user_sequences
    user_interacted: Dict[int, set] = model_service.user_interacted
    n_items = len(movie2idx)
    num_genres = model_service.config.get("num_genres", 20)

    if model_service.movie_genre_matrix is None or model_service.user_genre_matrix is None:
        raise SystemExit("Genre matrices not found — cannot evaluate.")

    # ---- popularity vector (interaction counts over model items) ----------
    pop = torch.zeros(n_items)
    for seq in user_sequences.values():
        for m in seq:
            idx = movie2idx.get(m)
            if idx is not None:
                pop[idx] += 1
    pop = pop.to(dev)

    all_movie_ids = list(movie2idx.keys())

    # ---- pick evaluation users --------------------------------------------
    users = [
        u for u, s in user_sequences.items()
        if len(s) >= args.min_history and u in user2idx and s[-1] in movie2idx
    ]
    if args.num_users and args.num_users > 0 and args.num_users < len(users):
        users = random.sample(users, args.num_users)
    print(f"Evaluating {len(users)} users (leave-one-out, 1 pos + {NUM_NEG} sampled negatives, K={K})")

    SYSTEMS = ["Popularity", "NCF Baseline", "NCF Hybrid", "Transformer", "Genre",
               "Static Ensemble", "Static + MMR", "DAMR (ours)"]
    if args.ablation:
        SYSTEMS += ["DAMR −adaptive", "DAMR −momentum", "DAMR −agreement",
                    "DAMR −quality", "DAMR −diversity"]

    agg = {name: {"HR": [], "NDCG": [], "MRR": [], "y": [], "p": []} for name in SYSTEMS}
    cov: Dict[str, set] = {name: set() for name in SYSTEMS}
    ilds: Dict[str, List[float]] = {name: [] for name in SYSTEMS}

    per_user_rows: List[dict] = []
    ablation_switches = {
        "DAMR −adaptive": {"use_adaptive": False},
        "DAMR −momentum": {"use_momentum": False},
        "DAMR −agreement": {"use_agreement": False},
        "DAMR −quality": {"use_quality": False},
        "DAMR −diversity": {"use_diversity": False},
    }

    t0 = datetime.now()
    n_evaluated = 0
    for n_done, u in enumerate(users):
        seq = user_sequences[u]
        pos = seq[-1]
        hist = seq[:-1]
        pos_idx = movie2idx[pos]

        seen = set(user_interacted.get(u, set())) | set(seq)
        negs: List[int] = []
        guard = 0
        while len(negs) < NUM_NEG and guard < NUM_NEG * 50:
            guard += 1
            m = random.choice(all_movie_ids)
            if m not in seen and m != pos:
                negs.append(m)
        if len(negs) < NUM_NEG:
            continue

        items = [pos] + negs
        item_idxs = torch.tensor([movie2idx[m] for m in items], dtype=torch.long)
        u_idx = user2idx[u]
        u_gvec = F.normalize(model_service.user_genre_matrix[u_idx].float(), dim=0).to(dev)

        s_pop = score_popularity(item_idxs, pop)
        s_ncf_base = score_ncf_baseline(u_idx, item_idxs)
        s_ncf_hyb = score_ncf_hybrid(u_idx, item_idxs, u_gvec)
        s_tr = score_transformer(hist, item_idxs)
        s_gen = score_genre(item_idxs, u_gvec)
        s_static = settings.NCF_WEIGHT * s_ncf_hyb + settings.TRANSFORMER_WEIGHT * s_tr + settings.GENRE_WEIGHT * s_gen

        raw_scores: Dict[str, torch.Tensor] = {
            "Popularity": s_pop,
            "NCF Baseline": s_ncf_base,
            "NCF Hybrid": s_ncf_hyb,
            "Transformer": s_tr,
            "Genre": s_gen,
            "Static Ensemble": s_static,
        }

        # ratings / counts for the pool (Bayesian quality prior)
        pool_ids = [model_service.idx2movie.get(int(i)) for i in item_idxs]
        meta = model_service.metadata_by_id  # populated at import from the data catalog
        ratings = [meta.get(mid, (None, None))[0] for mid in pool_ids]
        counts = [meta.get(mid, (None, None))[1] for mid in pool_ids]

        # DAMR state from the user's (synthetic-timestamped) history —
        # estimated once and shared by every Stage-4 variant
        profile = estimate_user_state(synth_history(hist), base_profile=u_gvec.cpu(), num_genres=num_genres)

        order_mmr, scores_mmr = rerank_variant(
            "mmr", item_idxs, s_ncf_hyb, s_tr, s_gen, profile, ratings, counts, K)
        order_damr, scores_damr = rerank_variant(
            "damr", item_idxs, s_ncf_hyb, s_tr, s_gen, profile, ratings, counts, K)

        rank_orders = {"Static + MMR": order_mmr, "DAMR (ours)": order_damr}
        rel_scores = {"Static + MMR": scores_mmr, "DAMR (ours)": scores_damr}

        if args.ablation:
            for name, over in ablation_switches.items():
                sw = dict(switches_for_variant("damr"))
                sw.update(over)
                full = damr_rerank(cand_idxs=item_idxs, s_ncf=s_ncf_hyb, s_tr=s_tr, s_gen=s_gen,
                                   profile=profile, ratings=ratings, counts=counts,
                                   top_k=len(item_idxs), **sw)
                rank_orders[name] = [e["pos"] for e in full]
                rel_map = {e["pos"]: e["score"] for e in full}
                # candidate-order alignment (index 0 = the positive) for AUC
                rel_scores[name] = [rel_map[i] for i in range(len(item_idxs))]

        # ---------------- record metrics ----------------
        n_evaluated += 1
        for name in SYSTEMS:
            y = [1] + [0] * NUM_NEG
            if name in raw_scores:
                s = raw_scores[name].detach().cpu()
                order_idx = torch.argsort(s, descending=True).tolist()
                scores_list = s.tolist()
            elif name in rank_orders:
                order_idx = rank_orders[name]
                scores_list = rel_scores[name]
            else:
                continue

            rank = order_idx.index(0) + 1  # position of the positive
            hit, ndcg, mrr = hr_ndcg_mrr(rank, K)
            agg[name]["HR"].append(hit)
            agg[name]["NDCG"].append(ndcg)
            agg[name]["MRR"].append(mrr)
            agg[name]["y"].extend(y)
            agg[name]["p"].extend(scores_list)
            ilds[name].append(intra_list_diversity(list(order_idx[:K])))
            cov[name].update(list(order_idx[:K]))

        if n_done % 200 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  {n_done}/{len(users)} users  ({elapsed:.0f}s)")

        # per-user row for segment analysis (static vs damr)
        static_rank = torch.argsort(raw_scores["Static Ensemble"].cpu(), descending=True).tolist().index(0) + 1
        damr_rank = rank_orders["DAMR (ours)"].index(0) + 1
        per_user_rows.append({
            "userId": u,
            "history_len": len(hist),
            "drift": profile.state.drift,
            "focus": profile.state.focus,
            "maturity": profile.state.maturity,
            "freshness": profile.state.freshness,
            "w_ncf": profile.state.weights[0],
            "w_tr": profile.state.weights[1],
            "w_gen": profile.state.weights[2],
            "static_rank": static_rank,
            "damr_rank": damr_rank,
            "static_hit": int(static_rank <= K),
            "damr_hit": int(damr_rank <= K),
            "static_ndcg": 1.0 / math.log2(static_rank + 1) if static_rank <= K else 0.0,
            "damr_ndcg": 1.0 / math.log2(damr_rank + 1) if damr_rank <= K else 0.0,
        })

    # ---------------- aggregate & save ----------------
    results: Dict[str, dict] = {}
    for name in SYSTEMS:
        a = agg[name]
        if not a["HR"]:
            continue
        res = {
            f"HR@{K}": round(float(np.mean(a["HR"])), 4),
            f"NDCG@{K}": round(float(np.mean(a["NDCG"])), 4),
            "MRR": round(float(np.mean(a["MRR"])), 4),
            "AUC": round(auc_from_scores(a["y"], a["p"]), 4),
            f"ILD@{K}": round(float(np.mean(ilds[name])), 4) if ilds[name] else None,
            f"Coverage@{K}": round(len(cov[name]) / n_items, 4),
        }
        results[name] = res

    payload = {
        "protocol": (
            f"Temporal leave-one-out: for each user the last interaction is the test positive, "
            f"combined with {NUM_NEG} uniformly sampled unseen negatives; rank of the positive "
            f"is evaluated at K={K}. NCF-paper style protocol (He et al., 2017)."
        ),
        "num_users_evaluated": n_evaluated,
        "num_negatives": NUM_NEG,
        "k": K,
        "seed": args.seed,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "num_users_total": len(model_service.user2idx),
            "num_items": n_items,
            "num_genres": num_genres,
        },
        "results": results,
    }

    out_json = Path(args.out) if args.out else Path(model_service.ml_dir) / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_csv = out_json.with_name(out_json.stem + "_by_user.csv")
    if per_user_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_user_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_user_rows)

    print(f"\nEvaluated {payload['num_users_evaluated']} users — leave-one-out, "
          f"1 pos + {NUM_NEG} neg, K={K}\n")
    header = f"{'Model':<16}{'HR@' + str(K):>8}{'NDCG@' + str(K):>10}{'MRR':>8}{'AUC':>8}{f'ILD@{K}':>9}{f'Cov@{K}':>9}"
    print(header)
    print("-" * len(header))
    for name in SYSTEMS:
        r = results.get(name)
        if not r:
            continue
        print(f"{name:<16}{r[f'HR@{K}']:>8.4f}{r[f'NDCG@{K}']:>10.4f}{r['MRR']:>8.4f}"
              f"{r['AUC']:>8.4f}{(r[f'ILD@{K}'] or 0):>9.4f}{r[f'Coverage@{K}']:>9.4f}")
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {out_csv}")


def _load_metadata() -> Dict[int, Tuple[Optional[float], Optional[int]]]:
    """movieId -> (rating, rating_count), lazily built from idx2movie + DB-free sources."""
    # ratings/counts live in PostgreSQL, but the evaluation must run offline.
    # Fallback: use the movie catalog JSON shipped in backend/data.
    meta: Dict[int, Tuple[Optional[float], Optional[int]]] = {}
    catalog = BACKEND_DIR / "data" / "movie_catalog.json"
    try:
        with open(catalog, "r", encoding="utf-8") as f:
            for m in json.load(f):
                meta[int(m["movieId"])] = (
                    float(m["rating"]) if m.get("rating") is not None else None,
                    int(m["rating_count"]) if m.get("rating_count") is not None else None,
                )
    except Exception:
        pass
    return meta


model_service.metadata_by_id = _load_metadata()  # type: ignore[attr-defined]

if __name__ == "__main__":
    main()
