# Filmory recommendation serving review — 13 October 2026

**Follow-up:** [14 October frozen-policy confirmation](CONFIRMATION_EVALUATION_2026-10-14.md) evaluates 1,000 additional users without changing the selected policy. Its aggregate intervals support a conditional replay gain. The mixed 200-user results below remain historical evidence and are not replaced.

## Delivered behavior

New registered accounts can now transition from the existing cold-start method to a sequence-and-neighbor recommendation path after at least two distinct mapped movie events. The path does not borrow an existing user's NCF embedding. It retrieves 300 eligible candidates with the frozen Transformer, then blends normalized collaborative-neighbor votes, Transformer scores and genre affinity. The weights depend on history length and were selected using validation NDCG@10.

The frontend no longer hard-codes a 100-candidate request for every account. The backend chooses the validated path's default. Already played, liked and saved titles and missing catalogue records are excluded before new-user candidate retrieval. Sequence ordering includes a stable event-ID tie-break; recent unmapped events do not immediately hide usable mapped events. Accounts with too little mapped activity, unavailable sequence artifacts or no policy overlay preserve the previous cold-start path.

The active model weights and their bundle configuration are unchanged. `backend/ml/serving_policy.json` is a separate, checkpoint-hash-bound overlay. Startup rejects policies for different checkpoints. Removing the overlay and restarting restores the old new-user default. Mapped training users retain their existing validated 100-candidate normalized fixed ensemble: the tested wider/mixed pools did not beat it on validation.

## Experiment design

1. Tested candidate budgets of 100, 300 and 1,000, Transformer and mixed retrieval, and fixed weight combinations using existing checkpoints. No training was performed.
2. Compared the sequence-only trial with legacy cold-start on long train-history replays. Legacy cold-start remained competitive; that trial was not used blindly as the new default.
3. Evaluated realistic short-history contexts of 2, 5, 10 and 20 events. Both legacy and new policies received the same visible history and genres, without a learned requester embedding. Tried Transformer, collaborative and mixed retrieval with predeclared weight choices. All four decisions were frozen in `selection.json` before final test evaluation.
4. Used 200 validation users and an additional 200 test users, excluding the earlier reported validation/test cohorts and the preceding candidate investigation's additional test users. The item catalogue contains 22,836 mapped items. Targets were excluded from fitting by the original clean split. Only the visible simulated account history was excluded from candidates.

These users and items were present in the training population. Hiding the requester embedding and truncating its history simulates a new account; it does not establish generalization to genuinely new people or prospective satisfaction.

## Measured results

All selected short-history policies use 300 Transformer-retrieved candidates. Weights below are **neighbor votes / Transformer / genre**, not NCF-user weights.

| Visible events | Selected weights | Validation HR@10 old → new | Validation NDCG@10 old → new | Additional test HR@10 old → new | Additional test NDCG@10 old → new |
|---|---|---|---|---|---|
| 2 | 0.65 / 0.25 / 0.10 | 6.0% → 7.5% | 0.031897 → 0.037354 | 2.0% → 2.5% | 0.010582 → 0.010422 |
| 5 | 0.80 / 0.20 / 0.00 | 7.5% → 9.0% | 0.034003 → 0.040762 | 2.0% → 3.0% | 0.009600 → 0.009217 |
| 10 | 0.80 / 0.20 / 0.00 | 6.0% → 8.0% | 0.026821 → 0.034147 | 4.0% → 4.0% | 0.016844 → 0.017475 |
| 20 | 0.40 / 0.50 / 0.10 | 7.0% → 9.0% | 0.030815 → 0.036394 | 4.5% → 4.0% | 0.019921 → 0.018645 |

Validation improved in every history group. Test results are mixed. Averaging the four contexts per test user gives a +0.25 percentage-point hit-rate change and a −0.000297 NDCG change. A paired bootstrap over the 200 users gives 95% intervals of −1.125 to +1.375 percentage points for HR@10 and −0.008646 to +0.005238 for NDCG. Both include zero. **This experiment does not demonstrate a statistically reliable overall recommendation-quality improvement.** It establishes working new-account model access and a validation-selected serving policy, with measured trade-offs.

The historical Transformer model-level 90.75% sampled HR@10 and 3.50% full-catalogue HR@10 remain unchanged. The new table evaluates a different serving path and cohort and must not be substituted for those paper results or called overall accuracy.

## Verification and artifacts

- Full backend suite: 80 passed. After adding an authenticated endpoint-transition regression and the final loader reset, the affected API/integration suites passed all 24 tests, including the new case.
- Frontend API/session suite: 6 passed. TypeScript and production client/server build passed; full lint had zero errors and eight existing fast-refresh warnings. Final model-page copy passed its lint and type check.
- Tests cover cold-to-sequence transition, actual HTTP registration/activity/recommendation flow, absence of borrowed requester embeddings, candidate uniqueness, empty pools, saved/missing-title exclusions, finite scores and rollback fallback.
- No real training run, checkpoint overwrite, PostgreSQL test-user mutation, or public deployment occurred. No preview server was needed or left running for this pass.

Evidence directories: `backend/evaluation/candidate_policies_20261013`, `backend/evaluation/new_user_policies_20261013`, and `backend/evaluation/new_user_collaborative_policies_20261013`. The final directory includes per-user rankings, validation selection, test metrics, a paired uncertainty result and source snapshots. Checkpoint SHA-256 values are recorded and verified unchanged. Initial experiment scripts were extended for the final collaborative trial; the final directory contains its source snapshots, while earlier protocol/choice details remain in their result files.

Execution logs: `docs/research/candidate-policies-20261013.txt`, `cold-start-transition-20261013.txt`, `new-user-policies-20261013.txt`, `new-user-collaborative-policies-20261013.txt`, `quality-backend-tests-20261013.txt`, `quality-final-targeted-20261013.txt`, `quality-frontend-checks-20261013.txt`, and `quality-frontend-lint-final-20261013.txt`. The initial frontend log contains a formatting-only lint error in the shortened API call; the formatter fixed it and the final full lint log supersedes that lint result.

## Remaining work for commercial claims

This pass does not certify Netflix/Hotstar-level performance or an error-free project. A larger prospective user study, impression/dismissal logging, multiple training seeds and a new timestamped evaluation are still needed to establish recommendation usefulness. Live providers, concurrency/failure recovery, controlled PostgreSQL migration reconciliation and streaming infrastructure require separate deployment validation. Human study participation and satisfaction evidence cannot be manufactured by software changes. The research LaTeX source was not edited in this serving experiment.
