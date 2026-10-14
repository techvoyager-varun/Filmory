# Frozen-policy confirmation — 14 October 2026

The current local new-account policy was evaluated on 1,000 additional users without retraining, changing its weights or selecting a new policy. This larger cohort supports a ranking improvement under the stated offline replay protocol. It does not establish human satisfaction or perfect recommendation accuracy.

## Results

| Visible history | Legacy HR@10 | Frozen policy HR@10 | Legacy NDCG@10 | Frozen policy NDCG@10 | Candidate recall@300 |
|---|---:|---:|---:|---:|---:|
| 2 events | 2.4% | 4.0% | 0.011920 | 0.017752 | 41.0% |
| 5 events | 3.3% | 4.4% | 0.015546 | 0.020171 | 46.6% |
| 10 events | 3.2% | 3.7% | 0.016437 | 0.017786 | 50.0% |
| 20 events | 4.8% | 5.1% | 0.023082 | 0.024218 | 51.3% |

After averaging each person's four contexts, mean HR@10 changes from 3.425% to 4.300%. The difference is +0.875 percentage points, with a 95% paired user-bootstrap interval of +0.325 to +1.475 percentage points. Mean NDCG difference is +0.003235524, with interval +0.001140612 to +0.005417095. Both intervals exclude zero. The intervals measure user-sampling uncertainty with these checkpoints; one training seed cannot establish variability across independent model fits.

## Protocol and scope

- The standalone serving policy and checkpoint hashes were recorded before outcomes were calculated.
- Cohort seed: 20261014. Earlier recorded training-run validation/test cohorts and exploratory evaluation cohorts were excluded: 2,191 identifiers. Cohort membership is retained in `protocol.json`.
- Universe: 22,836 mapped MovieLens items. Only the simulated account's visible history is excluded from candidate eligibility.
- Input: last 2, 5, 10 or 20 training-prefix interactions and genres computed from that context. Target: the original clean split's second withheld item.
- The requester has no learned NCF embedding in the evaluated path. Neighbor votes may use the train-only histories of the transductive training population; hiding the requester embedding does not make these genuinely unseen people.
- Baseline: the existing collaborative-neighbor algorithm. Comparison: the already selected 300-candidate Transformer retrieval with neighbor/sequence/genre fusion.
- Bootstrap: 10,000 percentile resamples, seed 20261015, resampling users after averaging their correlated contexts.
- The earlier 200-user result remains mixed and is reported in the paper. The confirmation was not used to retune the serving policy or replace the original 400-user sampled and 200-user model-level catalogue results.

## Evidence and reproduction

Artifacts are under `backend/evaluation/frozen_policy_20261014`: frozen protocol, cohort, source snapshots, source/checkpoint hashes, four per-user ranking files, final metrics and uncertainty estimates. The log is `docs/research/frozen-policy-confirmation-20261014.txt`.

From `E:/Filmory/backend`, use a new output directory:

```powershell
.venv/Scripts/python.exe scripts/evaluate_frozen_serving_policy.py --output evaluation/NEW_CONFIRMATION --users 1000 --seed 20261014
```

The evaluator excludes recorded earlier cohorts, including this confirmation after it has been saved. Therefore the command selects additional users on a later run. To reproduce the exact published cohort, use the saved `confirmation_users` and source snapshots; do not relabel a new cohort as the same experiment.

`research_evidence_verified_20261014.json` separately verifies all original sampled results by rerunning the fixed checkpoints, recomputes catalogue and ranking-policy aggregates from per-user rows, and checks that both targets are absent from every training history. Checkpoint hashes remain unchanged.

## Local implementation correction

The upstream live genre vector now respects the active bundle's L1-normalization declaration. The hybrid network already normalized its learned genre-projection input internally, so the earlier L2 upstream vector was not evidence of an uncorrected hybrid scoring-order bug. The upstream correction makes the preprocessing contract explicit and consistent. A regression compares serving input/predictions with the stored training profile and checks that dynamic genre updates remain L1-normalized. All 25 affected API/integration tests passed.

The `.tex` source has been updated in place, uses the supplied author/affiliation information, and compiled successfully using the built-in editor. No model retraining, public deployment, new PDF export, or preview server startup occurred.
