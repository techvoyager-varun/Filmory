# DAMR — Drift-Aware Momentum Re-Ranking for Multi-Expert Recommendation

**DAMR** is Filmory's Stage-4 re-ranking algorithm and the research contribution of this
project. It addresses a weakness of every fixed-weight ensemble recommender: *the relative
reliability of long-term vs. short-term taste signals is not constant — it changes per user
and per moment*, yet systems like Filmory's original Stage 1–3 pipeline fuse the experts
with static weights (`0.55·NCF + 0.25·Transformer + 0.20·Genre`) for every user, forever.

DAMR measures **how much a user's taste is currently changing**, and uses that measurement to

1. **re-weight the three experts per request** through a closed-form softmax gate over four
   interpretable user-state scalars, and
2. **score candidates along the direction the taste is moving** (a momentum term).

The fixed-weight ensemble is a *provable special case* of DAMR (Proposition 1), so the
method strictly generalises the system it replaces.

---

## 1. Position in the pipeline

```
Stage 1  NCF Hybrid           → candidate pool (K=100) + expert scores s_NCF
Stage 2  Sequential Transformer → expert scores s_TR
Stage 3  Genre affinity        → expert scores s_GEN
Stage 4  DAMR                  → user-state estimation
                                 → drift-adaptive expert fusion        (novel)
                                 → taste-momentum term                 (novel)
                                 → expert-agreement confidence         (novel)
                                 → Bayesian quality prior              (cited: IMDb)
                                 → MMR diversity selection             (cited: SIGIR'98)
                               → final top-10
```

Stages 1–3 only produce score vectors; **all fusion happens in Stage 4**. The previous
static ensemble remains available as `variant=static` for ablation and comparison.

## 2. Notation

| Symbol | Meaning |
|:---|:---|
| `g_i ∈ {0,1}^G` | binary genre vector of candidate `i` (from `movie_genre_matrix`, G = 20) |
| `s_NCF,i, s_TR,i, s_GEN,i` | expert scores of candidate `i` (min–max normalised over the pool) |
| `H = [(m_k, t_k)]_{k=1..n}` | user's interaction history, chronological (DB interactions; training-sequence fallback) |
| `G_long` | lifetime genre profile — uniform sum of `g_{m_k}`, L2-normalised |
| `G_short` | time-decayed profile — `Σ exp(−(t_now − t_k)/τ)·g_{m_k}`, τ = 7 days, L2-normalised |
| `M` | momentum vector, `M = G_short − G_long` (not normalised) |

## 3. Algorithm

### Step 1 — User-state estimation (the new signal)

```
drift      δ = 1 − cos(G_short, G_long)            how far current taste moved from identity
focus      φ = 1 − H(G_short) / log G              entropy: concentrated vs scattered interest
maturity   μ = log(1+n) / log(1+N_ref)             history depth (NCF reliability), N_ref = 200
freshness  f = exp(−hours_since_last / τ_s)        active-session signal, τ_s = 48 h
```

All four are cheap, closed-form, and interpretable — they are surfaced in the UI
(`GET /api/taste-state`, profile page "Taste Drift" card) and logged per user during
evaluation for segment analysis.

### Step 2 — Drift-adaptive expert weighting (replaces 0.55 / 0.25 / 0.20)

```
z_NCF = a₀ + a₁·μ − a₂·δ          more history → trust NCF; more drift → distrust it
z_TR  = b₀ + b₁·δ + b₂·f          more drift / active session → trust the sequence
z_GEN = c₀ + c₁·φ + c₂·(1−μ)      focused or new users → trust the genre signal
w = softmax(z_NCF, z_TR, z_GEN)
```

**Calibration (Proposition 1).** The intercepts are derived, not tuned:

```
a₀ = log(w*ₙ/ w*𝓰) − (a₁·μ̄ − a₂·δ̄)
b₀ = log(w*ₜ / w*𝓰) − (b₁·δ̄ + b₂·f̄)
c₀ = −(c₁·φ̄ + c₂·(1−μ̄))
```

where `w* = (0.55, 0.25, 0.20)` is the legacy weight vector and `(δ̄, φ̄, μ̄, f̄)` is a
calibration anchor (0.1, 0.4, 0.6, 0.5). Consequently `softmax(z(anchor)) = w*` **exactly**,
and with all slopes set to 0 the gate outputs `w*` for *every* state — i.e. static fusion is
DAMR with a degenerate gate. Unit-tested in `tests/test_damr.py`.

### Step 3 — Taste-momentum score

```
s_MOM,i = max(0, cos(g_i, M_unit))          (0 when ‖M‖ ≈ 0)
rel_i   = w·[s_NCF,i, s_TR,i, s_GEN,i]  +  η · δ · s_MOM,i        η = 0.3
```

NCF and the Transformer extrapolate from where the user *has been*; the momentum term is the
only component that extrapolates the *derivative* of taste. It self-extinguishes for stable
users (`δ ≈ 0 ⇒ bonus ≈ 0`), so DAMR intervenes only when it should.

### Step 4 — Expert-agreement confidence

```
agree_i = 1 − std(s_NCF,i, s_TR,i, s_GEN,i) / 0.5        ∈ [0, 1]
rel_i  ← rel_i · (1 − γ + γ · agree_i)                    γ = 0.2
```

Items all three experts endorse get a small bonus; items only one expert likes are damped.

### Step 5 — Quality prior + diversity *(standard components — cited, not claimed as novel)*

```
quality_i = [ v/(v+m) · R_i + m/(v+m) · C ] / 5            IMDb Bayesian rating, m=500, C=3.5
rel_i     ← (1−q_w)·rel_i + q_w·quality_i                  q_w = 0.15
sim(i,j)  = 0.5·cos(NCF item emb_i, emb_j) + 0.5·cos(g_i, g_j)     (same dual signal as /similar)
selection: greedy MMR, pick argmax λ·rel_i − (1−λ)·max_{j∈S} sim(i,j),  λ = 0.7, |S| → 10
```

### Pseudocode (Algorithm 1)

```
Input : candidates C (pool from Stages 1–3), expert scores s_NCF, s_TR, s_GEN,
        history H = [(movie, timestamp)], top_k = 10
Output: ranked list of top_k

 1  G_long  ← normalise( Σ_(m,t)∈H genre(m) )                       // Step 1
 2  G_short ← normalise( Σ_(m,t)∈H exp(−(now−t)/τ) · genre(m) )
 3  δ ← 1 − cos(G_short, G_long);   φ ← 1 − H(G_short)/log G
 4  μ ← log(1+|H|)/log(1+N_ref);    f ← exp(−(now−t_last)/τ_s)
 5  M ← G_short − G_long
 6  w ← softmax([a₀+a₁μ−a₂δ,  b₀+b₁δ+b₂f,  c₀+c₁φ+c₂(1−μ)])        // Step 2
 7  for each candidate i ∈ C:                                       // Steps 3–5
 8      s_MOM,i ← max(0, cos(genre(i), M/‖M‖))
 9      rel_i   ← w·[s_NCF,i, s_TR,i, s_GEN,i] + η·δ·s_MOM,i
10      agree_i ← 1 − std(s_NCF,i, s_TR,i, s_GEN,i)/0.5
11      rel_i   ← rel_i · (1−γ + γ·agree_i)
12      rel_i   ← (1−q_w)·rel_i + q_w·BayesianQuality(i)
13 return MMR(C, rel, λ, top_k)
```

### Complexity

State estimation is `O(n + G)`; fusion is `O(K)`; the MMR loop is `O(K²)` vectorised with
K = 100 → **1–3 ms per request on CPU**. No training, no retraining, no new model artifacts.

## 4. Variants / ablation switches

`damr_rerank(...)` exposes one boolean per step: `use_adaptive`, `use_momentum`,
`use_agreement`, `use_quality`, `use_diversity`. Predefined variants:

| Variant | Switches | What it represents |
|:---|:---|:---|
| `damr` (default) | all on | the proposed method |
| `mmr` | adaptive/momentum/agreement off | static fusion + quality + diversity (engineering baseline) |
| `static` | (legacy path) | the original fixed-weight ensemble, byte-compatible behaviour |

Request them with `GET /api/recommendations/{userId}?variant=damr|mmr|static` — this is also
how the live demo compares lists side by side. The `--ablation` flag of
`scripts/evaluate.py` additionally evaluates leave-one-component-out DAMR.

## 5. What is ours vs. cited

| Component | Status |
|:---|:---|
| Drift/focus/maturity/freshness state estimation from decayed-vs-uniform genre profiles | **ours** |
| Closed-form drift-conditioned expert gate with static fusion as a special case | **ours** |
| Momentum term `η·δ·cos(genre, G_short − G_long)` | **ours** |
| Expert-agreement confidence factor | **ours** (simple, but new in this combination) |
| Bayesian quality prior | IMDb weighted-rating formula — cite |
| MMR diversity | Carbonell & Goldstein, SIGIR 1998 — cite |
| NCF / SASRec backbone | He et al. 2017 / Kang & McAuley 2018 — cite |

## 6. Offline evaluation protocol

Implicit-feedback recommenders are **not** measured with classification accuracy (≈99.9 % of
user–item pairs are negatives). The standard protocol (NCF paper, SASRec) is *temporal
leave-one-out*:

- for each user, hold out their **last** interaction as the test positive;
- sample **99** uniformly-drawn unseen movies as negatives;
- rank the 100 items and record where the positive landed.

Reported metrics: **HR@K, NDCG@K, MRR, AUC** (accuracy) and **ILD@K, catalogue coverage**
(beyond-accuracy). Known protocol caveats, stated honestly in the report:

- uniform negatives are easier for popularity — the Popularity baseline is reported
  precisely to make this visible;
- the shipped checkpoints were trained on all interactions (including the held-out one), so
  the learned models' numbers are optimistic; a strictly clean number requires retraining on
  the leave-one-out split.

Reproduce:

```bash
cd backend
python scripts/evaluate.py --num-users 2000            # main table → ml/metrics.json
python scripts/evaluate.py --num-users 500 --ablation \
       --out ml/metrics_ablation.json                  # ablation table
python -m pytest tests/test_damr.py -v                 # unit tests (no artifacts needed)
```

`ml/metrics_by_user.csv` logs per-user state (δ, φ, μ, f, gate weights, static vs DAMR
ranks) — the input for the paper's segment analysis (gains by drift quartile).

## 7. API surface added for Stage 4

| Endpoint | Purpose |
|:---|:---|
| `GET /api/recommendations/{userId}?variant=damr\|mmr\|static` | choose the re-ranker per request |
| `GET /api/metrics` | serves `ml/metrics.json` (the evaluation table above) |
| `GET /api/taste-state` | live δ/φ/μ/f, gate weights, G_long/G_short/momentum vectors (radar chart) |
