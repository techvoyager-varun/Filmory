"""
Stage 4 — DAMR: Drift-Aware Momentum Re-Ranker
==============================================

The novel contribution of Filmory. DAMR sits after the three scoring stages
(NCF Hybrid, Sequential Transformer, Genre affinity) and answers two questions
the fixed ensemble `0.55*NCF + 0.25*TR + 0.20*Genre` cannot:

1. *How much should each expert be trusted right now, for this user, in this
   moment?*  — answered by an explicit taste-drift estimate that re-weights the
   three experts per request through a closed-form softmax gate.
2. *Where is this user's taste heading?* — answered by a momentum vector
   (time-decayed profile minus lifetime profile) that gives a score bonus to
   candidates lying in the direction the taste is moving.

Method (one pass, no training required):

  Step 1  Estimate the user state from their interaction history:
            drift      delta = 1 - cos(G_short, G_long)
            focus      phi   = 1 - H(G_short) / log(num_genres)
            maturity   mu    = log(1+n) / log(1+N_ref)
            freshness  f     = exp(-hours_since_last / tau_session)
          and the momentum vector M = G_short - G_long.

  Step 2  Drift-adaptive expert weighting (replaces the fixed 0.55/0.25/0.20):
            z_NCF = a0 + a1*mu - a2*delta
            z_TR  = b0 + b1*delta + b2*f
            z_GEN = c0 + c1*phi + c2*(1-mu)
            w = softmax(z_NCF, z_TR, z_GEN)
          The intercepts a0/b0/c0 are derived from a calibration anchor so that
          an "average" user receives exactly the classic weights — i.e. the old
          static ensemble is a special case of DAMR (all slopes zero, or the
          anchor state itself).

  Step 3  Taste-momentum score:
            s_MOM(i) = max(0, cos(genre(i), M))
            rel(i)  += eta * delta * s_MOM(i)

  Step 4  Expert-agreement confidence:
            agree(i) = 1 - std(s_NCF, s_TR, s_GEN) / 0.5
            rel(i)  *= (1 - gamma + gamma * agree(i))

  Step 5  Quality prior + diversity (standard components, cited, not claimed
          as novel):
            quality(i) = IMDb-style Bayesian weighted rating / 5
            rel(i)     = (1 - qw) * rel(i) + qw * quality(i)
            MMR greedy selection with sim(i,j) = 0.5*cos(NCF item emb)
                                                  + 0.5*cos(genre vectors)

Every step can be switched off independently for ablation studies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from app.config import settings
from app.ml.model_service import model_service


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class UserState:
    """The four interpretable scalars DAMR estimates for a user, plus the
    expert weights that were derived from them."""

    drift: float = 0.0        # delta — how far current taste moved from identity
    focus: float = 0.0        # phi   — how concentrated the current interest is
    maturity: float = 0.0     # mu    — how much history exists (NCF reliability)
    freshness: float = 0.0    # f     — is the user in an active session
    n_interactions: int = 0
    weights: Tuple[float, float, float] = (
        settings.NCF_WEIGHT,
        settings.TRANSFORMER_WEIGHT,
        settings.GENRE_WEIGHT,
    )  # filled after fusion: (ncf, transformer, genre)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "drift": round(self.drift, 4),
            "focus": round(self.focus, 4),
            "maturity": round(self.maturity, 4),
            "freshness": round(self.freshness, 4),
            "nInteractions": self.n_interactions,
        }
        d["weights"] = {
            "ncf": round(self.weights[0], 4),
            "transformer": round(self.weights[1], 4),
            "genre": round(self.weights[2], 4),
        }
        return d


@dataclass
class TasteProfile:
    """Full result of Step 1 — state scalars, genre profiles and momentum."""

    state: UserState
    g_long: torch.Tensor      # (num_genres,) lifetime genre profile (L2-normalised)
    g_short: torch.Tensor     # (num_genres,) time-decayed genre profile (L2-normalised)
    momentum: torch.Tensor    # (num_genres,) M = G_short - G_long (NOT normalised)

    def vectors_to_lists(self) -> Dict[str, List[float]]:
        r = lambda t: [round(float(x), 4) for x in t.tolist()]  # noqa: E731
        return {"long": r(self.g_long), "short": r(self.g_short), "momentum": r(self.momentum)}


# ---------------------------------------------------------------------------
# Step 1 — user-state estimation
# ---------------------------------------------------------------------------
def _entropy(p: torch.Tensor) -> float:
    """Shannon entropy of a non-negative vector treated as a distribution."""
    total = float(p.sum())
    if total <= 0:
        return 0.0
    probs = p / total
    probs = probs[probs > 0]
    return float(-(probs * probs.log()).sum())


def _to_naive_utc(ts: datetime) -> datetime:
    """DB stores naive UTC timestamps; drop tzinfo if a tz-aware dt sneaks in."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def estimate_user_state(
    history: Sequence[Tuple[int, datetime]],
    base_profile: Optional[torch.Tensor] = None,
    now: Optional[datetime] = None,
    num_genres: Optional[int] = None,
) -> TasteProfile:
    """
    Estimate the user's taste state from a chronological interaction history.

    Parameters
    ----------
    history : list of (model_item_index, timestamp), oldest -> newest.
    base_profile : optional (num_genres,) tensor with the user's training-time
        genre profile (row of `user_genre_matrix`). Blended into the lifetime
        profile so that a mapped user's on-platform history does not throw away
        what the model already knows about them.
    now : reference time (defaults to UTC now).
    """
    genre_matrix = model_service.movie_genre_matrix
    if genre_matrix is None:
        # Degenerate environment (no artifacts): neutral state.
        g = torch.zeros(num_genres or 20)
        return TasteProfile(UserState(), g, g.clone(), torch.zeros_like(g))

    G = genre_matrix.float()
    num_genres = num_genres or G.shape[1]
    now = _to_naive_utc(now) if now is not None else datetime.utcnow()
    n = len(history)

    g_long_raw = torch.zeros(num_genres, device=G.device)
    g_short_raw = torch.zeros(num_genres, device=G.device)
    for item_idx, ts in history:
        if item_idx < 0 or item_idx >= G.shape[0]:
            continue
        gvec = G[item_idx]
        g_long_raw += gvec
        age_days = max(0.0, (now - _to_naive_utc(ts)).total_seconds() / 86400.0)
        g_short_raw += math.exp(-age_days / settings.DAMR_TAU_DAYS) * gvec

    if base_profile is not None:
        bp = base_profile.detach().to(G.device).float()
        if bp.dim() != 1 or bp.shape[0] != num_genres:
            bp = bp.flatten()[:num_genres]
        # Blend the training-time profile with weight proportional to the amount
        # of on-platform history (a brand-new account keeps its full seed).
        g_long_raw += F.normalize(bp, dim=0) * max(1, n) * 0.5

    # Lifetime profile — uniform mean direction. Falls back to the decayed one
    # (and vice versa) so that drift stays 0 when nothing is observable.
    if g_long_raw.sum() > 0:
        G_long = F.normalize(g_long_raw, dim=0)
    else:
        G_long = F.normalize(g_short_raw, dim=0) if g_short_raw.sum() > 0 else g_long_raw
    if g_short_raw.sum() > 0:
        G_short = F.normalize(g_short_raw, dim=0)
    else:
        G_short = G_long.clone()

    # -- the four state scalars -------------------------------------------
    if G_long.sum() > 0 and G_short.sum() > 0:
        cos = float(torch.dot(G_short, G_long).clamp(-1.0, 1.0))
        drift = 1.0 - cos
    else:
        drift = 0.0  # nothing observable yet — never claim drift
    focus = 1.0 - _entropy(G_short) / math.log(num_genres) if num_genres > 1 and G_short.sum() > 0 else 0.0
    focus = max(0.0, focus)
    maturity = min(1.0, math.log1p(n) / math.log1p(max(2, settings.DAMR_N_REF)))
    if n:
        hours_since_last = max(0.0, (now - _to_naive_utc(history[-1][1])).total_seconds() / 3600.0)
        freshness = math.exp(-hours_since_last / settings.DAMR_TAU_SESSION_H)
    else:
        freshness = 0.0

    state = UserState(
        drift=drift,        # keep full precision — the momentum gate tests drift > 0
        focus=focus,
        maturity=maturity,
        freshness=freshness,
        n_interactions=n,
    )
    momentum = G_short - G_long
    return TasteProfile(state, G_long, G_short, momentum)


# ---------------------------------------------------------------------------
# Step 2 — drift-adaptive expert weighting
# ---------------------------------------------------------------------------
def compute_logit_base() -> Tuple[float, float, float]:
    """
    Derive the intercepts (a0, b0, c0) so that at the calibration anchor state
    the softmax gate produces EXACTLY the classic fixed weights
    (0.55 / 0.25 / 0.20). This makes the static ensemble a special case of
    DAMR — the property the paper's Proposition 1 relies on.

    Given the anchor state and slopes, the logits are:
        z_NCF = a0 + a1*mu  - a2*delta
        z_TR  = b0 + b1*delta + b2*f
        z_GEN = c0 + c1*phi + c2*(1-mu)
    Requiring softmax(z) = w* forces:
        z_NCF - z_GEN = log(w_ncf / w_genre)
        z_TR  - z_GEN = log(w_tr   / w_genre)
    Fixing the free level at z_GEN(anchor) = 0 gives closed-form intercepts.
    """
    s = settings.DAMR_SLOPES
    anchor = settings.DAMR_ANCHOR
    d, phi, mu, f = anchor["drift"], anchor["focus"], anchor["maturity"], anchor["freshness"]
    w_ncf, w_tr, w_gen = settings.NCF_WEIGHT, settings.TRANSFORMER_WEIGHT, settings.GENRE_WEIGHT

    z_ncf_anchor = s["a1"] * mu - s["a2"] * d
    z_tr_anchor = s["b1"] * d + s["b2"] * f
    z_gen_anchor = s["c1"] * phi + s["c2"] * (1.0 - mu)

    a0 = math.log(w_ncf / w_gen) - z_ncf_anchor
    b0 = math.log(w_tr / w_gen) - z_tr_anchor
    c0 = -z_gen_anchor
    return (round(a0, 6), round(b0, 6), round(c0, 6))


_LOGIT_BASE: Optional[Tuple[float, float, float]] = None


def adaptive_weights(state: UserState) -> torch.Tensor:
    """Closed-form drift-conditioned gate over the three experts."""
    global _LOGIT_BASE
    if _LOGIT_BASE is None:
        _LOGIT_BASE = compute_logit_base()
    a0, b0, c0 = _LOGIT_BASE
    s = settings.DAMR_SLOPES
    z = torch.tensor(
        [
            a0 + s["a1"] * state.maturity - s["a2"] * state.drift,
            b0 + s["b1"] * state.drift + s["b2"] * state.freshness,
            c0 + s["c1"] * state.focus + s["c2"] * (1.0 - state.maturity),
        ],
        dtype=torch.float32,
    )
    return torch.softmax(z, dim=0)


# ---------------------------------------------------------------------------
# Step 5 helpers — quality prior, item-item similarity, MMR
# ---------------------------------------------------------------------------
def bayesian_quality(rating: Optional[float], count: Optional[int]) -> float:
    """IMDb-style Bayesian weighted rating, normalised to [0, 1]."""
    m = settings.BAYES_M
    C = settings.BAYES_C
    R = float(rating) if rating else C
    v = float(count) if count else 0.0
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    return wr / 5.0


def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max to [0,1]; a constant vector maps to the neutral midpoint 0.5."""
    lo, hi = x.min(), x.max()
    if float(hi - lo) <= 0:
        return torch.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def item_pair_similarity(item_idxs: Sequence[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Pairwise item similarity used by Stage 4's diversity term — the SAME
    dual-signal similarity the /similar/{movieId} endpoint uses:
        sim(i, j) = 0.5 * cos(NCF item embeddings) + 0.5 * cos(genre vectors)
    Returns an (n, n) tensor on `device`.
    """
    device = device or model_service.device
    with torch.no_grad():
        idx = torch.as_tensor(list(item_idxs), dtype=torch.long, device=device)

        sim = torch.zeros(len(item_idxs), len(item_idxs), device=device)
        emb = model_service.ncf_hybrid.item_embedding.weight[idx] if model_service.ncf_hybrid is not None else None
        if emb is not None:
            emb_n = F.normalize(emb.float(), dim=1)
            sim += 0.5 * (emb_n @ emb_n.T)

        G = model_service.movie_genre_matrix
        if G is not None:
            gen_n = F.normalize(G[idx].float(), dim=1)
            sim += 0.5 * (gen_n @ gen_n.T)
        else:
            sim += 0.5 * torch.eye(len(item_idxs), device=device)
        return sim.clamp(-1.0, 1.0)


def intra_list_diversity(item_idxs: Sequence[int]) -> float:
    """ILD = average pairwise (1 - sim) within a list, diagonal excluded.
    Higher = more diverse."""
    n = len(item_idxs)
    if n < 2:
        return 0.0
    sim = item_pair_similarity(item_idxs)
    one_minus = 1.0 - sim
    one_minus.fill_diagonal_(0.0)
    return float(one_minus.sum() / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Main entry point — Stage 4 re-ranking
# ---------------------------------------------------------------------------
@torch.no_grad()
def damr_rerank(
    cand_idxs: torch.Tensor,             # (K,) model item indices of the candidate pool
    s_ncf: torch.Tensor,                 # (K,) Stage 1 NCF Hybrid scores
    s_tr: torch.Tensor,                  # (K,) Stage 2 Transformer scores
    s_gen: torch.Tensor,                 # (K,) Stage 3 genre-affinity scores
    profile: TasteProfile,
    ratings: Sequence[Optional[float]],  # length K — average ratings (or None)
    counts: Sequence[Optional[int]],     # length K — rating counts (or None)
    top_k: int = None,                   # final list size (default settings.TOP_K)
    *,
    use_adaptive: bool = True,           # Step 2 — drift-adaptive expert gate
    use_momentum: bool = True,           # Step 3 — taste-momentum bonus
    use_agreement: bool = True,          # Step 4 — expert-agreement confidence
    use_quality: bool = True,            # Step 5a — Bayesian quality prior
    use_diversity: bool = True,          # Step 5b — MMR diversity selection
) -> List[Dict[str, Any]]:
    """
    Re-rank the candidate pool and return the final top-k as a list of dicts:

        { "pos", "modelIdx", "score" (final DAMR relevance), "ncfScore",
          "transformerScore", "genreScore", "momentumScore",
          "agreementScore", "qualityScore", "diversityPenalty",
          "mmrScore" }
    ordered best-first. `profile.state.weights` is updated in place with the
    gate output so callers can surface it in the UI.
    """
    top_k = top_k or settings.TOP_K
    device = model_service.device
    K = len(cand_idxs)
    if K == 0:
        return []

    S = torch.stack(
        [_minmax(s_ncf.to(device)), _minmax(s_tr.to(device)), _minmax(s_gen.to(device))],
        dim=1,
    )  # (K, 3) — experts put on a common [0,1] scale

    # ---- Step 2: drift-adaptive fusion ----------------------------------
    if use_adaptive:
        w = adaptive_weights(profile.state).to(device)
    else:
        w = torch.tensor(
            [settings.NCF_WEIGHT, settings.TRANSFORMER_WEIGHT, settings.GENRE_WEIGHT],
            device=device,
        )
    profile.state.weights = (float(w[0]), float(w[1]), float(w[2]))
    rel = S @ w  # (K,)

    # ---- Step 3: taste-momentum bonus ------------------------------------
    momentum = profile.momentum.to(device)
    s_mom = torch.zeros(K, device=device)
    if use_momentum and float(momentum.norm()) > 1e-6 and profile.state.drift > 0:
        m_unit = F.normalize(momentum, dim=0)
        if model_service.movie_genre_matrix is not None:
            gvecs = F.normalize(model_service.movie_genre_matrix[cand_idxs.to(device)].float(), dim=1)
            s_mom = (gvecs @ m_unit).clamp(min=0.0)
        rel = rel + settings.DAMR_ETA * profile.state.drift * s_mom

    # ---- Step 4: expert-agreement confidence ------------------------------
    if use_agreement:
        agreement = 1.0 - S.std(dim=1) / 0.5
        agreement = agreement.clamp(0.0, 1.0)
        rel = rel * (1.0 - settings.DAMR_GAMMA + settings.DAMR_GAMMA * agreement)
    else:
        agreement = torch.ones(K, device=device)

    # ---- Step 5a: Bayesian quality prior ----------------------------------
    if use_quality:
        q = torch.tensor(
            [bayesian_quality(r, c) for r, c in zip(ratings, counts)],
            dtype=torch.float32,
            device=device,
        )
        rel = (1.0 - settings.QUALITY_WEIGHT) * rel + settings.QUALITY_WEIGHT * q
    else:
        q = torch.zeros(K, device=device)

    # ---- Step 5b: MMR diversity-aware selection ---------------------------
    if use_diversity and K > 1:
        sim = item_pair_similarity(cand_idxs.tolist(), device)
        lam = settings.MMR_LAMBDA
        chosen: List[int] = []
        remaining = set(range(K))
        penalties: Dict[int, float] = {}
        while len(chosen) < min(top_k, K) and remaining:
            best_i, best_v = None, -1e9
            # Vectorised: max similarity of each remaining item to the chosen set
            if chosen:
                sel = torch.tensor(chosen, device=device)
                rem = torch.tensor(sorted(remaining), device=device)
                max_sims = sim.index_select(0, rem).index_select(1, sel).max(dim=1).values
                mmr_vals = lam * rel[rem] - (1.0 - lam) * max_sims
                local = int(torch.argmax(mmr_vals))
                best_i = int(rem[local])
                best_v = float(mmr_vals[local])
                penalties[best_i] = float(max_sims[local])
            else:
                for i in remaining:
                    v = float(rel[i])
                    if v > best_v:
                        best_i, best_v = i, v
                penalties[best_i] = 0.0
            chosen.append(best_i)
            remaining.discard(best_i)
    else:
        chosen = torch.topk(rel, k=min(top_k, K)).indices.tolist()
        penalties = {i: 0.0 for i in chosen}

    state_dict = profile.state.to_dict()
    out: List[Dict[str, Any]] = []
    for i in chosen:
        out.append(
            {
                "pos": i,
                "modelIdx": int(cand_idxs[i]),
                "score": round(float(rel[i]), 4),
                "ncfScore": round(float(s_ncf[i]), 4),
                "transformerScore": round(float(s_tr[i]), 4),
                "genreScore": round(float(s_gen[i]), 4),
                "momentumScore": round(float(s_mom[i]), 4),
                "agreementScore": round(float(agreement[i]), 4),
                "qualityScore": round(float(q[i]), 4),
                "diversityPenalty": round(penalties.get(i, 0.0), 4),
                "expertWeights": state_dict["weights"],
                "userState": state_dict,
            }
        )
    return out


VARIANT_SWITCHES: Dict[str, Dict[str, bool]] = {
    # Pure Stage 1-3 fixed ensemble (the original system, exact behaviour).
    "static": {
        "use_adaptive": False, "use_momentum": False, "use_agreement": False,
        "use_quality": False, "use_diversity": False,
    },
    # Static fusion + quality prior + MMR diversity (the engineering baseline).
    "mmr": {
        "use_adaptive": False, "use_momentum": False, "use_agreement": False,
        "use_quality": True, "use_diversity": True,
    },
    # Full DAMR (the proposed method).
    "damr": {
        "use_adaptive": True, "use_momentum": True, "use_agreement": True,
        "use_quality": True, "use_diversity": True,
    },
}


def switches_for_variant(variant: str) -> Dict[str, bool]:
    return VARIANT_SWITCHES.get(variant, VARIANT_SWITCHES["damr"])
