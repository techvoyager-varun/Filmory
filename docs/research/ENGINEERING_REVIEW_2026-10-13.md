# Filmory engineering review — 13 October 2026

The identified defects below are corrected and the executed checks pass. This is evidence of improved functionality, not a certification that every input, deployment or recommendation will be correct.

## Changes

- Authentication now validates restored sessions with the server, handles expired tokens, and prevents a late response from an old account or a cancelled sign-in from replacing the current session. HTTP errors preserve useful validation messages.
- Saved-list/history/like request failures propagate to the UI. Actions report failures, guard against duplicate submissions, and ignore responses belonging to a different active user. Taste query caching is scoped to the user.
- Assistant requests ignore stale responses after changing conversations. Session ownership, feedback references, blank messages and filter names are validated. Per-user rate-limit rejections no longer consume shared IP quota.
- The offline parser handles ordinary genre requests, multiple excluded genres, fractional-hour runtime caps, and common constraint-clearing follow-ups. Discovery no longer requires genre/runtime/rating/year filter words to occur in a title or synopsis; clearing a filter cannot leave those words acting as hidden restrictions. Missing synopses do not automatically disqualify metadata matches.
- Template explanations cover every displayed movie. Generated explanations must reference every supplied movie exactly once with valid evidence references. This reference validation does not prove every generated sentence is factual.
- Cold-start fallback excludes supplied favorite/history IDs and considers all supplied favorite genres. Similarity retrieval handles small catalogues without requesting an oversized top-k or returning the reference item.
- Catalogue browsing has stable tie-breaking and null-safe ordering. Literal search text does not act as SQL wildcard syntax. Genre rails match whole genre labels.
- Registration creates the account and its preferences atomically; blank names, unknown favorite genres and missing favorite movies are rejected before preferences change.
- API timestamps include UTC offsets. Database connection and pool waits are bounded. A new migration supplies the four assistant tables missing from the original migration, and fresh migrations are tested against the application schema.
- The taste panel identifies adaptive weights as a diagnostic rather than claiming they are the default home ranking. The popularity rail no longer claims weekly activity. History uses placeholder artwork for missing posters, shows a loading state, and identifies playback as simulated instead of showing an invented progress bar.

## Verification

| Check | Result |
|---|---|
| Full backend suite | 76 passed; 5 warnings |
| Final targeted backend check after filter-removal correction | 35 passed; 2 dependency warnings |
| Frontend API/session regression suite | 6 passed |
| TypeScript | Passed |
| Frontend lint | 0 errors; 8 existing fast-refresh warnings |
| Production client and server build | Passed |
| Final history component type/lint check | Passed |
| Python dependency consistency | No broken requirements |
| Active baseline, hybrid and Transformer checkpoint SHA-256 checks | All match the saved run manifest |

Backend API tests and browser writes use a disposable SQLite catalogue populated from the active bundle's title/genre metadata. Ratings, runtimes and descriptions in that fixture are explicitly synthetic and are **not research results**. Active checkpoint inference is exercised; the user's PostgreSQL account/activity records are not changed by these tests. The existing tiny training-continuation unit test uses temporary toy models only.

Browser verification: demo sign-in; constrained conversational recommendations; correct ten-film explanation count; runtime-filter removal; saving a recommendation to My List; liking it; recording a simulated play; session restoration and persisted history after a full page load. Screenshot: `history-review-20261013.png`. This is a smoke check in the in-app browser, not an exhaustive device/browser matrix.

Logs: `backend-review-20261013.txt`, `final-targeted-review-20261013.txt`, `frontend-review-20261013.txt`, and `history-final-check-20261013.txt`.

## Model quality and remaining limits

The active bundle remains `backend/training/runs/20261001_clean_extended`. No additional real training was run and no trained checkpoint was changed. Previously measured selected BCE remains approximately 0.189611 for the hybrid and 0.136147 for the Transformer. The Transformer previously achieved 90.75% sampled test HR@10 but only 3.50% full-catalogue test HR@10; the hybrid catalogue figure was 3.00%. These are held-out ranking measurements, not overall accuracy or guarantees that a person will enjoy a recommendation. This maintenance review does not establish a new ranking-quality improvement.

Home recommendations use the active bundle's validation-selected default policy. The conversational assistant separately uses its configured DAMR reranker after metadata candidate retrieval. Heuristic parsing and keyword search remain limited for nuanced natural-language requests.

Live Gemini/TMDB calls were deliberately disabled in the disposable browser test. External-provider reliability, live generated factuality, public deployment, concurrent load, every mobile/browser layout, and new model-quality experiments remain unverified in this pass. Playback is a simulation, not a streaming service.

The existing PostgreSQL database has an unstamped migration history from earlier `create_all` usage. The new migration verifies fresh installations; do not blindly run or stamp it over existing tables. Back up and compare the existing schema before reconciling migration history for deployment. No destructive migration or blind stamping was performed.

No Turnitin/plagiarism certification or journal acceptance assessment was performed in this engineering pass. The research source and trained model artifacts are preserved.
