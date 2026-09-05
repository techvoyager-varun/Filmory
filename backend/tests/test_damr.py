"""
Unit tests for Stage 4 — DAMR (Drift-Aware Momentum Re-Ranker).

Runs against tiny synthetic artifacts injected into the ModelService singleton,
so no pickles / .pth files / PostgreSQL are required:

    cd backend && python -m pytest tests/test_damr.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml import damr
from app.ml.damr import (
    UserState,
    adaptive_weights,
    bayesian_quality,
    compute_logit_base,
    damr_rerank,
    estimate_user_state,
    intra_list_diversity,
    _minmax,
)
from app.ml.model_service import model_service
from app.config import settings


# ---------------------------------------------------------------------------
# Fixtures — inject a tiny synthetic "model service" environment
# ---------------------------------------------------------------------------
NUM_ITEMS = 12
NUM_GENRES = 4
REAL_SLOPES = {"a1": 0.8, "a2": 1.5, "b1": 2.0, "b2": 0.6, "c1": 0.8, "c2": 1.0}


def setup_module(module=None):
    """Give the singleton a small genre matrix + a stub NCF item embedding."""
    torch.manual_seed(0)

    genre_matrix = torch.zeros(NUM_ITEMS, NUM_GENRES)
    for i in range(NUM_ITEMS):
        genre_matrix[i, i % NUM_GENRES] = 1.0
        genre_matrix[i, (i + 1) % NUM_GENRES] = 0.5
    model_service.movie_genre_matrix = genre_matrix
    model_service.device = torch.device("cpu")

    class _Emb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(NUM_ITEMS, 8) * 0.1)

    model_service.ncf_hybrid = type("Stub", (), {"item_embedding": _Emb()})()


def teardown_module(module=None):
    model_service.movie_genre_matrix = None
    model_service.ncf_hybrid = None
    damr._LOGIT_BASE = None


def _profile(history, base_profile=None):
    damr._LOGIT_BASE = None
    return estimate_user_state(history, base_profile=base_profile, num_genres=NUM_GENRES)


def _cands(n=NUM_ITEMS):
    idxs = torch.arange(n)
    torch.manual_seed(1)
    return idxs, torch.rand(n), torch.rand(n), torch.rand(n)


# ---------------------------------------------------------------------------
# Proposition 1 — static fusion is a special case of DAMR
# ---------------------------------------------------------------------------
def test_anchor_state_reproduces_fixed_weights():
    """At the calibration anchor the softmax gate must output exactly
    the classic 0.55 / 0.25 / 0.20 weights."""
    a0, b0, c0 = compute_logit_base()
    anchor = settings.DAMR_ANCHOR
    s = settings.DAMR_SLOPES
    z = torch.tensor([
        a0 + s["a1"] * anchor["maturity"] - s["a2"] * anchor["drift"],
        b0 + s["b1"] * anchor["drift"] + s["b2"] * anchor["freshness"],
        c0 + s["c1"] * anchor["focus"] + s["c2"] * (1 - anchor["maturity"]),
    ])
    w = torch.softmax(z, dim=0)
    assert torch.allclose(w, torch.tensor([0.55, 0.25, 0.20]), atol=1e-6)


def test_zero_slopes_reduce_to_static_weights_for_any_state():
    """With every slope = 0 the gate ignores the state entirely and returns
    the fixed weights — i.e. DAMR generalises static fusion."""
    settings.DAMR_SLOPES = {k: 0 for k in REAL_SLOPES}
    damr._LOGIT_BASE = None
    try:
        w = adaptive_weights(UserState(drift=0.9, focus=0.9, maturity=0.1, freshness=0.9))
        assert torch.allclose(w, torch.tensor([0.55, 0.25, 0.20]), atol=1e-6)
    finally:
        settings.DAMR_SLOPES = dict(REAL_SLOPES)
        damr._LOGIT_BASE = None


# ---------------------------------------------------------------------------
# Step 1 — state estimation
# ---------------------------------------------------------------------------
def test_stable_user_has_zero_drift_and_zero_momentum():
    """A user who always interacts with the same genre has G_short == G_long."""
    now = datetime.utcnow()
    history = [(0, now - timedelta(days=k)) for k in range(150, 4, -1)]
    p = _profile(history)
    assert p.state.drift < 1e-4
    assert p.momentum.norm() < 1e-4
    assert p.state.maturity > 0.9     # long history -> mature
    assert p.state.freshness < 0.1    # last watch is 5 days old


def test_drifting_user_gets_drift_and_momentum_toward_new_genre():
    """Long history of genre 0, then a burst of genre 2 in the last 2 days:
    drift must be large, momentum must point at genre 2, and the gate must
    shift weight toward the sequential expert."""
    now = datetime.utcnow()
    history = [(0, now - timedelta(days=d)) for d in range(200, 7, -1)]
    history += [(2, now - timedelta(hours=24 - k * 5)) for k in range(5)]  # 5 watches in ~24h
    p = _profile(history)
    assert p.state.drift > 0.2
    assert int(torch.argmax(p.g_short)) == 2
    assert int(torch.argmax(p.momentum)) == 2
    w = adaptive_weights(p.state)
    assert w[1] > 0.25, "a hard-drifting, active user should trust the sequence expert more"
    assert p.state.freshness > 0.9


def test_base_profile_keeps_training_signal_for_new_accounts():
    """An account with no on-platform history still inherits its mapped
    training profile -> zero drift but a meaningful lifetime profile."""
    base = torch.zeros(NUM_GENRES)
    base[1] = 5.0
    p = _profile([], base_profile=base)
    assert p.state.n_interactions == 0
    assert p.state.drift == 0.0
    assert int(torch.argmax(p.g_long)) == 1


def test_empty_user_is_neutral():
    p = _profile([])
    assert p.state.drift == 0.0
    assert p.g_long.sum() == 0


# ---------------------------------------------------------------------------
# Steps 2-5 — re-ranking behaviour
# ---------------------------------------------------------------------------
def test_switches_off_equals_plain_weighted_sum_ordering():
    idxs, s_ncf, s_tr, s_gen = _cands()
    p = _profile([])
    out = damr_rerank(
        idxs, s_ncf, s_tr, s_gen, p,
        ratings=[3.0] * NUM_ITEMS, counts=[100] * NUM_ITEMS,
        top_k=5, use_adaptive=False, use_momentum=False,
        use_agreement=False, use_quality=False, use_diversity=False,
    )
    expected = (
        0.55 * _minmax(s_ncf) + 0.25 * _minmax(s_tr) + 0.20 * _minmax(s_gen)
    )
    expected_order = torch.argsort(expected, descending=True)[:5].tolist()
    assert [e["pos"] for e in out] == expected_order


def test_momentum_boosts_candidates_in_drift_direction():
    """User drifting toward genre 2: a genre-2 movie and everything else share
    identical expert scores — with momentum on, the genre-2 movie must win."""
    now = datetime.utcnow()
    history = [(0, now - timedelta(days=d)) for d in range(150, 7, -1)]
    history += [(2, now - timedelta(hours=3))]
    p = _profile(history)

    n = 8
    idxs = torch.arange(n)
    s = torch.full((n,), 0.5)
    kwargs = dict(use_diversity=False, use_agreement=False, use_quality=False)

    out_on = damr_rerank(idxs, s, s.clone(), s.clone(), p, [None] * n, [None] * n,
                         top_k=n, use_momentum=True, **kwargs)
    out_off = damr_rerank(idxs, s, s.clone(), s.clone(), p, [None] * n, [None] * n,
                          top_k=n, use_momentum=False, **kwargs)
    assert out_on[0]["pos"] == 2, "the drift-target genre should be ranked first"
    assert out_on[0]["momentumScore"] > 0.4
    assert all(e["momentumScore"] == 0.0 for e in out_off)
    assert all(e["score"] == e_off["score"] for e, e_off in zip(out_on, out_off)) is False


def test_agreement_prefers_consensus_items():
    """Item 0 is loved by ALL three experts; item 1 has the same fused score
    before Step 4 but is contested by one expert. With agreement on, item 0
    must come first; with agreement off they tie in fused-score order."""
    idxs = torch.arange(3)  # item 2 is a low-score filler for min-max scale
    p = _profile([])
    s_ncf = torch.tensor([0.9, 0.9, 0.45])
    s_tr = torch.tensor([0.9, 0.1, 0.5])
    s_gen = torch.tensor([0.9, 0.9, 0.45])
    kw = dict(use_adaptive=False, use_momentum=False, use_quality=False, use_diversity=False)
    out_on = damr_rerank(idxs, s_ncf, s_tr, s_gen, p, [None] * 3, [None] * 3, top_k=3,
                         use_agreement=True, **kw)
    out_off = damr_rerank(idxs, s_ncf, s_tr, s_gen, p, [None] * 3, [None] * 3, top_k=3,
                          use_agreement=False, **kw)
    assert out_on[0]["pos"] == 0
    assert out_on[0]["agreementScore"] > out_on[1]["agreementScore"]
    # the consensus item keeps its score; the contested item is penalised
    assert out_on[0]["score"] == out_off[0]["score"]
    assert out_on[1]["score"] < out_off[1]["score"]


def test_quality_prior_prefers_high_ratings():
    """With equal expert scores the quality prior alone decides the order."""
    n = 6
    idxs = torch.arange(n)
    p = _profile([])
    s = torch.full((n,), 0.7)
    ratings = [4.8 if i % 2 == 0 else 2.0 for i in range(n)]
    counts = [10_000_000] * n  # many votes -> quality converges to raw rating
    out = damr_rerank(idxs, s, s.clone(), s.clone(), p, ratings, counts, top_k=3,
                      use_adaptive=False, use_momentum=False,
                      use_agreement=False, use_diversity=False, use_quality=True)
    assert all(e["pos"] % 2 == 0 for e in out), "high-rated movies should dominate"


def test_diversity_avoids_near_duplicates():
    """10 near-identical genre-0 movies + 2 genre-2 movies whose relevance is
    only slightly lower: pure relevance fills the list with duplicates; MMR
    must trade a duplicate slot for a diverse item."""
    n = 12
    idxs = torch.arange(n)
    genre_matrix = torch.zeros(n, NUM_GENRES)
    genre_matrix[:10, 0] = 1.0          # near-duplicates
    genre_matrix[10:, 2] = 1.0          # the only diverse items
    model_service.movie_genre_matrix = genre_matrix
    try:
        p = _profile([])
        # after min-max: dups ≈ (1.0, 1.0, 0.5) -> rel 0.9; item 10 ≈ (0, 1.0, 0.5)
        # -> rel 0.35; item 11 ≈ (1.0, 0, 0.5) -> rel 0.65 — clearly below the dups
        s_ncf = torch.tensor([0.9] * 10 + [0.89, 0.9])
        s_tr = torch.tensor([0.9] * 10 + [0.9, 0.89])
        s_gen = torch.tensor([0.9] * 12)
        common = dict(use_adaptive=False, use_momentum=False,
                      use_agreement=False, use_quality=False)
        out_div = damr_rerank(idxs, s_ncf, s_tr, s_gen, p,
                              [None] * n, [None] * n, top_k=5, use_diversity=True, **common)
        out_pure = damr_rerank(idxs, s_ncf, s_tr, s_gen, p,
                               [None] * n, [None] * n, top_k=5, use_diversity=False, **common)
        assert all(e["pos"] < 10 for e in out_pure), "pure relevance = all duplicates"
        assert any(e["pos"] >= 10 for e in out_div), "MMR must reserve a slot for a diverse item"
        assert intra_list_diversity([e["pos"] for e in out_div]) > \
               intra_list_diversity([e["pos"] for e in out_pure])
    finally:
        setup_module()


def test_intra_list_diversity_behaviour():
    assert intra_list_diversity([0]) == 0.0
    same = intra_list_diversity([0, 0])    # identical items -> zero diversity
    mixed = intra_list_diversity([0, 2])   # different genres -> more diverse
    assert same == 0.0
    assert mixed > same


def test_bayesian_quality():
    assert bayesian_quality(None, None) == settings.BAYES_C / 5.0
    # many votes: converges to the raw rating
    assert abs(bayesian_quality(4.0, 10_000_000) - 4.0 / 5.0) < 1e-3
    # few votes: pulled toward the global prior
    assert abs(bayesian_quality(5.0, 5) - settings.BAYES_C / 5.0) < 0.05
