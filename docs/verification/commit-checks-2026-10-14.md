# Pre-commit verification — 14 October 2026

Scope: the local Filmory application fixes, recommendation policies and feedback,
optional classmate review, training/evaluation utilities, and their recorded
verification evidence. This check did not retrain models, retune ranking weights,
submit the manuscript, push to a remote repository, or restart the separate
architecture explorer.

## Current checks

- Backend: `python -m pytest tests --basetemp .test-tmp/<unique-run> -q` — **100 passed** in 91.72 seconds. Tests use disposable databases and disable external AI/poster calls. Five warnings concern existing deprecated library/test APIs and PyTorch prototype nested tensors.
- Frontend: `npm run test` — **9 passed**.
- Frontend: `npm run typecheck` — passed.
- Frontend: `npm run lint` — zero errors, eight existing Fast Refresh warnings.
- Frontend: `npm run build` — client and server builds passed. Existing nonfatal notices concern tsconfig-path plugin redundancy and an ineffective dynamic import.
- Tracked patch whitespace: `git diff --check` — no whitespace errors. Git emitted Windows line-ending conversion notices.
- Staged application patch whitespace passed. Two trailing-space lines in the frozen historical architecture snapshot are retained to preserve its evidence hash; `.gitattributes` prevents newline conversion for frozen training/evaluation artifacts.
- Targeted credential pattern scan of application, test, verification and helper sources found no Google API key, GitHub token or private-key blocks. This is a limited local scan, not a security certification. Local `.env` files are excluded.
- Exact staged-content audit: 197 files, about 12.8 MB; 41 JSON files, 26 JSONL files and 71 Python files parsed successfully. No excluded runtime binaries, unexpectedly large files or targeted credential patterns were found. Selected frozen artifact bytes match the working evidence.
- Active bundle integrity: all 27 entries in its SHA-256 manifest matched the local files, including checkpoints and reproducibility metadata.

## Recommendation evidence

The latest browser check covered five individual genre preferences and five
successive plays from different genres. Each homepage list matched the API;
all played titles were excluded from Recommended For You. The cold-start fix
preserves all explicit genres. The final mixed-history slate favoured
Comedy/Animation and did not represent Sci-Fi/Documentary; no perfect-quality or
balanced-slate claim is made. See `five-genre-frontend-2026-10-14.md` and its JSON
and screenshot artifacts.

Offline pipeline results and source snapshots are retained separately from
functional checks. The frozen serving defaults were not replaced after the
latest validation-selected alternatives failed the held-out promotion rule.
Classmate participation remains pending; QA interactions are not human study
results.

## Repository scope

Source code, tests, verification helpers, reports, selected reproducibility
metadata, summary results and held-out result rows are suitable for the local
commit. Secrets, virtual environments, dependencies, build output, local
databases, large training binaries and large intermediate validation dumps are
excluded. The active trained bundle remains local and must be restored or
recreated on a different checkout, as explained in `backend/training/README.md`.

The research manuscript, notebook attachments, paused architecture explorer and
unselected historical/intermediate artifacts are preserved in the working tree
for their own review; they are not implicitly approved for publication by this
commit.
