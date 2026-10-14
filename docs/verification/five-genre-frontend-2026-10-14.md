# Homepage recommendations: five-genre frontend verification

Verified on 14 October 2026 using the local frontend at port 5173 and API at port 8100. All UI actions used a dedicated `Filmory Frontend QA` fixture, not an existing user's account. No likes, dislikes, saved films, or model retraining were needed for the play tests. Playback is simulated in Filmory: Play records an interaction; this is not a test of actual video viewing or completion tracking.

## Individual genre preferences

Each case completed onboarding through the frontend, selected one genre and no favourite films, and inspected the 12 cards in Recommended For You. An independent authenticated API check compared the exact ordered UI movie IDs with the live recommendations and verified full catalogue genres.

| Selected genre | Matching cards | UI/API order agreed |
| --- | --- | --- |
| Sci-Fi | 12/12 | Yes |
| Comedy | 12/12 | Yes |
| Thriller | 12/12 | Yes |
| Animation | 12/12 | Yes |
| Documentary | 12/12 | Yes |

These are 60 successful functional genre checks, not an accuracy or user satisfaction measurement. Compact cards display only the first two genres, so full catalogue metadata was used to verify genre matches.

## Five preferences and five plays

The same fixture then selected all five genres. This exposed a real cold-start bug: the dynamic top-three genre signal replaced explicit preferences. The fix in `backend/app/ml/recommender.py` retains every explicit genre and adds inferred recent interests. A regression test now asserts all five reach the cold-start function. The affected API and recommendation integration suites passed: **33 tests** (two upstream library warnings).

The mixed-preference feed was captured after the fix, before plays. Its historical snapshot is in the UI evidence file. Its API equality check occurred after the first play and therefore failed against changed state; it is not counted as a verified live comparison or as a stale-cache defect. All five subsequent after-play snapshots were independently compared while history was held unchanged.

Played films were Twelve Monkeys (Sci-Fi), Groundhog Day (Comedy), Seven/Se7en (Thriller), Toy Story (Animation), and Bowling for Columbine (Documentary). The featured carousel advanced before the first Play, so persisted history confirmed Twelve Monkeys as the actual first title.

| Distinct recorded plays | Displayed cards | Match at least one selected genre | Played titles excluded | UI/API order agreed |
| --- | --- | --- | --- | --- |
| 1 | 12 | 12/12 | Yes | Yes |
| 2 | 12 | 12/12 | Yes | Yes |
| 3 | 12 | 12/12 | Yes | Yes |
| 4 | 12 | 11/12 | Yes | Yes |
| 5 | 12 | 11/12 | Yes | Yes |

All lists were nonempty and unique. Scores were finite and bounded between zero and one. With two or more distinct mapped plays, every returned card reported `sequence_only` and no NCF user score, confirming history-based serving for a new account without borrowing a trained user's identity. Recommendations changed between each recorded state and appeared on Home through ordinary navigation/cache invalidation.

## Quality conclusion

The homepage works after playing across five genres. The final slate included Willy Wonka, Shrek, The Princess Bride, Toy Story 2, Monsters Inc., Monty Python and the Holy Grail, Dogma, Pirates of the Caribbean, Mission Impossible, Scream, Who Framed Roger Rabbit, and The Fellowship of the Ring.

The final slate represented Comedy, Animation and Thriller, but **no Sci-Fi or Documentary**. It also included one film outside all five preferred genres (The Fellowship of the Ring). Preferences are soft signals, not mandatory genre filters. The result favours overlapping Comedy/Animation/Fantasy interests; therefore this check does not establish ideal genre balance or best relevance for the person. Any proposed balancing policy should be compared with the frozen current policy on validation data and then held-out full-catalogue HR@10/NDCG@10 before replacing serving defaults. Human feedback remains necessary to judge perceived relevance. Functional tests do not change the previously reported offline metrics.

No model checkpoint, training setup, loss target, or validated ranking weights were changed in this verification. No paid service was used. The temporary QA account and its owned rows are removed after evidence capture; existing users, catalogue and model artifacts are preserved.

## Evidence

- `five-genre-ui-cases-20261014.json`: browser-read ordered cards, including mixed-preference and after-play snapshots.
- `five-genre-live-checks-20261014.json`: 10 successful comparisons (five individual genre cases and five after-play cases), full movie metadata and played-title exclusions.
- `five-genre-home-after-play-20261014.png`: visible final homepage.
- `five-genre-cleanup-20261014.json`: narrow QA fixture cleanup results.
- `tools/recommendation-verification/verify_genres.py`: repeatable, dedicated-QA-only comparison helper; run from backend with its virtualenv Python, supplying `--email`, `--ui-file` and `--report`.
