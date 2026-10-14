# Filmory project audit and training results

**Latest reviews:** [13 October engineering review](ENGINEERING_REVIEW_2026-10-13.md), [recommendation serving experiments](RECOMMENDATION_QUALITY_REVIEW_2026-10-13.md), [14 October confirmation](CONFIRMATION_EVALUATION_2026-10-14.md), and [current paper audit](PAPER_AUDIT.md). The report below records the original 1 October training/audit; its old test counts and serving behavior are superseded where the newer reviews say so. Trained model measurements remain unchanged.

1 October 2026. This is an engineering audit with executed local checks, not a guarantee that every feature or future deployment is perfect.

## Outcome

Corrected the training/serving contracts, trained fresh checkpoints with targets removed before fitting, activated a separate new bundle locally, corrected recommendation and assistant bugs, and updated the research draft with measured results. Original weights and the supplied original notebook remain preserved.

The active pointer is `backend/ml/active_bundle.json`, selecting `backend/training/runs/20261001_clean_extended`. The baseline and hybrid have 16-dimensional embeddings. The Transformer uses dedicated padding, last-valid pooling and a 256-dimensional feed-forward block. Checkpoints maximize validation NDCG, rather than minimize training loss. The hybrid selected epoch is nine even though epoch ten has lower BCE.

| Model | First-epoch BCE | Selected checkpoint BCE | Sampled test HR@10 | Full-catalogue test HR@10 |
|---|---:|---:|---:|---:|
| NCF baseline | 0.437975 | 0.209859 | 82.25% | 2.50% |
| NCF hybrid | 0.209473 | 0.189611 | 86.00% | 3.00% |
| Transformer | 0.302811 | 0.136147 | 90.75% | 3.50% |

BCE is the classification part of the objective; total optimized loss also includes a 0.25-weight pairwise term. Hybrid initialization comes from the baseline fitted in this clean run. These losses cannot be directly compared to the original notebook's different sampling/objective/budget. Sampled HR uses 400 users and one target plus 99 unique negatives. Full-catalogue HR uses 200 users, 22,836 mapped items and training-history exclusion. Neither is overall classification accuracy or a calibrated movie-match probability.

Transformer full-catalogue recall at 100 is 35.5%; its full-catalogue NDCG@10 is 0.018740. Catalogue-wide quality remains weak despite strong sampled ranking. Further tuning must use validation; a new untouched test population is needed for later final experiments.

## Ranking policy

The Transformer wins model-level catalogue validation NDCG and provides the candidate pool. On identical top-100 candidates, normalized fixed fusion, direct Transformer and DAMR are compared without a quality prior. Validation NDCG selects normalized fixed fusion as the default personalized policy. DAMR remains available explicitly.

| Policy | Validation NDCG@10 | Test NDCG@10 | Test HR@10 |
|---|---:|---:|---:|
| Direct Transformer | 0.028270 | 0.018740 | 3.50% |
| Normalized fixed | 0.041254 | 0.020308 | 2.50% |
| DAMR | 0.032382 | 0.015778 | 3.50% |

This tradeoff does not establish that the selected policy is best for every metric or user. The assistant's constrained-pool reranking and new-user content path remain separate behaviors, not covered by these mapped-user policy metrics.

## Issues corrected

- Hybrid feature-block order differed between notebook and serving despite compatible parameter shapes. Restored `[user, user genres, item, item genres]` and L1 genre input normalization.
- Transformer padding, one-based candidate shifts and cosine scoring differed from training. Restored zero-based IDs, padding equal to item count, dot-product scoring, and explicit legacy pooling compatibility. New training pools the last valid position and handles empty sequences safely.
- The original sequential training prefix cap favored early users. New training samples users uniformly with fresh prefix targets, clips gradients, reduces learning rate on plateaus and saves validation-selected checkpoints.
- Old evaluations held out previously trained-on interactions. Historical harnesses now refuse fresh bundles; clean metrics retain a saved pre-training split.
- Canonical MovieLens IDs were confused with model indices in history lookup and cold-start neighbors. Corrected conversions, including valid item/user index zero.
- Unmapped/new users borrowed a trained user's embedding. They now use their own preference/content path. Arbitrary assistant candidate pools retain original positions and metadata for unmapped items.
- Stored ordinal histories were assigned invented timestamps. They now use explicitly positional state estimates.
- Removed likes and watchlist entries continued to count as permanently excluded interactions. Exclusion now reflects current states plus viewing history. Unlike and onboarding changes rebuild live genre signals instead of retaining old boosts. Nonexistent delete targets return 404.
- Offline assistant extraction treated excluded genres as preferred, missed plural westerns and discarded rating floors. Corrected those cases; incomplete catalogue metadata no longer causes response validation failures.
- Provider extraction state used shared mutable flags. Request context now isolates extraction status, and fallback notices reflect the actual path. Configured provider timeout is applied. Invalid explanation evidence references fall back to templates; semantic claim support is still not proven.
- Backend omitted its Google SDK dependency. Added it. Zero-duration access-token overrides now expire correctly, and bcrypt input byte limits are validated.
- Frontend API parsing failed on successful 204 deletes; corrected. Account switches reset cached user data and discard stale responses. Play/like changes invalidate recommendation and taste queries. Onboarding handles failed saves without leaving a stuck button.
- Movie cards called uncalibrated scores match percentages. They now display scores. Model transparency separates sampled and catalogue results and shows actual fresh metrics.
- Frontend formatting/type/lint issues were corrected. Most formatting edits came from the existing ESLint/Prettier rules; there are eight remaining fast-refresh warnings.
- Health checks now verify database connectivity and loaded models and return 503 when unavailable. Fresh bundles reject incomplete learned artifacts.

## Executed verification

- Backend suite: 59 passing tests. New checks compare class predictions with reviewed original notebook class definitions, test pooling and padding, verify mapping/cold-start/pool contracts, parser constraints, byte-length/expiry behavior and preference changes.
- Training integrity: all 41,547 users' two targets are absent from their training histories; validation/test users are disjoint; all 800 evaluation profiles match train-only reconstruction. Training and architecture snapshots match recorded SHA-256 hashes. Training interactions: 5,923,646.
- Frontend TypeScript check passes. Production client/SSR build passes. Lint has zero errors and eight fast-refresh warnings.
- `pip check`: no broken requirements. `alembic current` executes, but prints no stamped revision for the local database. Migration history requires reconciliation before a controlled production deployment; no destructive migration or blind stamping was performed.
- Offline production assistant checks: 36 inspected queries, 198 returned movies, zero request errors, correct routing for all requests, zero hard-constraint violations after fixes versus 31 returned-instance violations before. The same checks pass against the active fresh bundle in `backend/evaluation/production_check_active`. Four clarification and eight expected-empty cases pass. Evidence references are valid; this is template fallback, not live generated explanation verification.
- Browser smoke: demo sign-in reaches the home recommendations, score labels render, and the model page visibly matches fresh sampled and full-catalogue results. Screenshot: `model-page-verified.png`. Automated backend tests cover assistant/session/API behavior; a complete manual test of every UI state and production browser has not been performed.

Warnings remain for deprecated naive UTC date calls, the test client's dependency transition, and PyTorch nested-tensor API status. They are not suppressed or represented as test failures.

## Evidence and reproduction

The corrected standalone notebook is `backend/training/filmory_training_v2.ipynb`. The exact supplied original is retained as `backend/training/source_notebook.ipynb`; `E:/dl-based-recommendersystem.ipynb` was not edited. The standalone notebook embeds architecture, preparation and training source, can use original MovieLens CSVs, and has no fabricated execution outputs.

The active bundle retains `metrics.json`, `training_log.jsonl`, `provenance.json`, `split.json`, sampled candidate arrays, per-user catalogue/policy rows, original run source snapshots and a SHA-256 manifest. Large model/data files are ignored by Git and remain local: a deployment or another checkout must receive the whole bundle before enabling its pointer. Restoring the original bundle requires removing the active pointer and restarting the backend; existing original weights remain in `backend/ml`.

From `E:/Filmory/backend`, reproduce the extended fitting budget into a NEW output directory:

```powershell
.venv/Scripts/python.exe scripts/train_models.py --output training/runs/NEW_RUN --epochs 10 --sequence-epochs 5 --positives-per-epoch 1500000 --sequence-samples 120000 --embedding-dim 16 --test-user-offset 800 --eval-users 400 --catalogue-users 200 --threads 4
.venv/Scripts/python.exe scripts/check_ranking_policy.py --bundle training/runs/NEW_RUN
```

These commands use stored ordinal histories. For timestamped data, use `prepare_training_data.py` with the original ratings/movies CSVs, then train with `--source` pointing to that isolated prepared source. Existing output directories are rejected to preserve experiments. For exact historical reproduction, use the stored source snapshots and recorded environment; later reviewed source includes extra snapshot/provenance safeguards.

## Research and remaining limits

`filmory_paper.tex` is revised around the actual fresh results, including the weak catalogue ranking and DAMR comparison. `PAPER_AUDIT.md` supersedes the earlier audit; the old diagnostic aggregates are preserved separately. The original notebook establishes MovieLens 20M selection and original latest-rating holdout; the earlier statement that it trained all raw interactions is corrected.

The new experiment uses one seed, small evaluation sets, bounded training budgets, ordinal histories and a static training prefix for both holdouts. The original notebook test targets and timestamps are unavailable locally. The 36 assistant queries have been inspected; neither set is an independent blinded test. Live provider quality, generated claim-level factuality, human utility, full deployment/load testing and multi-seed robustness remain unverified. Zero training loss is not a reasonable guarantee of generalization.

An exact Springer/IEEE journal, author/institution/advisor information, and a stronger novelty/effectiveness study remain needed before submission. Turnitin was not run, and no plagiarism score or publication guarantee is asserted. The native LaTeX compiler failed to initialize; `.tex` source is saved/opened, but compiled PDF layout is unverified.
