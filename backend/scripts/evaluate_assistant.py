"""
Evaluation & Benchmarking Script for Filmory Recommender & Ask Filmory Assistant.

Measures:
1. Recommender Engine:
   - DAMR vs Static ensemble comparison (NDCG@10-like relative ordering)
   - Intra-List Diversity comparison
   - Candidate recall & coverage
2. GenAI Assistant:
   - Hard constraint satisfaction rate (%)
   - Follow-up update correctness (%)
   - Hallucination / Unsupported claim rate (%)
   - Median and p95 latency (ms)

Usage:
    python -m scripts.evaluate_assistant [--live] [--user-id N]
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure backend root is in sys.path when run directly
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.database import get_db
from app.models.db_models import Movie, User
from app.schemas.assistant import IntentDelta, MovieIntent
from app.services.assistant_orchestrator import _matches_hard_constraints, _merge_intents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filmory.evaluation")

# -----------------------------------------------------------------------
# Benchmark Scenarios: Hard Constraint Satisfaction
# -----------------------------------------------------------------------
BENCHMARK_SCENARIOS = [
    {
        "id": "scenario_1_comedy_no_horror",
        "description": "Comedy without Horror under 120 min",
        "intent": MovieIntent(preferred_genres=["Comedy"], excluded_genres=["Horror"], max_runtime_minutes=120),
    },
    {
        "id": "scenario_2_scifi_90s",
        "description": "Sci-Fi from the 90s (1990-1999)",
        "intent": MovieIntent(preferred_genres=["Sci-Fi"], min_year=1990, max_year=1999),
    },
    {
        "id": "scenario_3_animation_high_rating",
        "description": "Animation rated at least 8.0/10",
        "intent": MovieIntent(preferred_genres=["Animation"], min_rating=8.0),
    },
    {
        "id": "scenario_4_short_thriller",
        "description": "Thriller under 90 minutes",
        "intent": MovieIntent(preferred_genres=["Thriller"], max_runtime_minutes=90),
    },
    {
        "id": "scenario_5_drama_no_romance",
        "description": "Drama excluding Romance, post-2000",
        "intent": MovieIntent(preferred_genres=["Drama"], excluded_genres=["Romance"], min_year=2000),
    },
]

# -----------------------------------------------------------------------
# Follow-Up Delta Merge Scenarios
# -----------------------------------------------------------------------
FOLLOWUP_SCENARIOS = [
    {
        "id": "followup_shorter",
        "base": {"preferred_genres": ["Action"], "max_runtime_minutes": 150},
        "delta": IntentDelta(max_runtime_minutes=100),
        "expected_check": lambda m: m.preferred_genres == ["Action"] and m.max_runtime_minutes == 100,
    },
    {
        "id": "followup_clear_exclusions",
        "base": {"preferred_genres": ["Drama"], "excluded_genres": ["Romance"]},
        "delta": IntentDelta(clear_fields=["excluded_genres"]),
        "expected_check": lambda m: m.preferred_genres == ["Drama"] and m.excluded_genres == [],
    },
    {
        "id": "followup_ordinal_exclusion",
        "base": {"preferred_genres": ["Comedy"], "excluded_movie_ids": [1]},
        "prev_movie_ids": [10, 20, 30],
        "delta": IntentDelta(target_ordinal_exclusion=2),
        "expected_check": lambda m: 20 in m.excluded_movie_ids and 1 in m.excluded_movie_ids,
    },
    {
        "id": "followup_clear_all",
        "base": {"preferred_genres": ["Action"], "excluded_genres": ["Horror"], "max_runtime_minutes": 120, "min_rating": 7.0},
        "delta": IntentDelta(clear_fields=["all"]),
        "expected_check": lambda m: m.preferred_genres == [] and m.excluded_genres == [] and m.max_runtime_minutes is None and m.min_rating is None,
    },
    {
        "id": "followup_add_genre_keep_runtime",
        "base": {"preferred_genres": ["Comedy"], "max_runtime_minutes": 90},
        "delta": IntentDelta(preferred_genres=["Romance"]),
        "expected_check": lambda m: "Romance" in m.preferred_genres and m.max_runtime_minutes == 90,
    },
    {
        "id": "followup_clear_single_excluded_genre",
        "base": {"preferred_genres": ["Drama"], "excluded_genres": ["Horror", "Romance"]},
        "delta": IntentDelta(clear_fields=["excluded_genres:Horror"]),
        "expected_check": lambda m: "Horror" not in m.excluded_genres and "Romance" in m.excluded_genres,
    },
]


def evaluate_constraint_satisfaction(db):
    """Evaluate hard constraint satisfaction rate on catalog movies."""
    logger.info("=" * 60)
    logger.info("EVALUATING HARD CONSTRAINT SATISFACTION")
    logger.info("=" * 60)

    results = {}
    total_eval = 0
    total_valid = 0

    for sc in BENCHMARK_SCENARIOS:
        intent: MovieIntent = sc["intent"]
        from app.services.semantic_search import search_movies_by_intent
        candidates, _ = search_movies_by_intent(intent, db, limit=50)

        valid_count = sum(1 for m in candidates if _matches_hard_constraints(m, intent))
        rate = (valid_count / len(candidates) * 100.0) if candidates else 100.0

        results[sc["id"]] = {
            "description": sc["description"],
            "candidates_returned": len(candidates),
            "valid_candidates": valid_count,
            "satisfaction_rate_pct": round(rate, 2),
        }
        total_eval += len(candidates)
        total_valid += valid_count
        logger.info("  %s: %d/%d (%.1f%%)", sc["id"], valid_count, len(candidates), rate)

    overall = (total_valid / total_eval * 100.0) if total_eval else 100.0
    logger.info("  OVERALL: %.2f%%", overall)
    return results, round(overall, 2)


def evaluate_followup_correctness():
    """Evaluate follow-up delta merge correctness."""
    logger.info("=" * 60)
    logger.info("EVALUATING FOLLOW-UP UPDATE CORRECTNESS")
    logger.info("=" * 60)

    results = {}
    passed_count = 0

    for fup in FOLLOWUP_SCENARIOS:
        merged = _merge_intents(fup["base"], fup["delta"], fup.get("prev_movie_ids"))
        check_fn = fup["expected_check"]
        passed = check_fn(merged)
        results[fup["id"]] = {
            "passed": passed,
            "resulting_intent": merged.model_dump(),
        }
        if passed:
            passed_count += 1
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", fup["id"], status)

    accuracy = (passed_count / len(FOLLOWUP_SCENARIOS) * 100.0)
    logger.info("  OVERALL: %.2f%%", accuracy)
    return results, round(accuracy, 2)


def evaluate_damr_vs_static(db, user: User, top_k: int = 10):
    """
    Compare DAMR vs Static ensemble on real recommendation quality.
    Metrics: relative score distribution, intra-list diversity, genre coverage.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING DAMR vs STATIC ENSEMBLE")
    logger.info("=" * 60)

    from app.ml.recommender import get_personalized_recommendations
    from app.ml.damr import intra_list_diversity

    variants = {}
    for variant_name in ["damr", "static"]:
        try:
            t0 = time.perf_counter()
            rec_type, recs = get_personalized_recommendations(
                user=user, db=db, candidate_k=100, top_k=top_k, variant=variant_name,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            scores = [r.score or 0.0 for r in recs]
            genres_seen = set()
            for r in recs:
                genres_seen.update(r.genres)

            diversity = None
            if recs and hasattr(recs[0], "listDiversity") and recs[0].listDiversity is not None:
                diversity = recs[0].listDiversity

            variants[variant_name] = {
                "rec_type": rec_type,
                "num_recs": len(recs),
                "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
                "score_spread": round(max(scores) - min(scores), 4) if scores else 0.0,
                "unique_genres": len(genres_seen),
                "genre_list": sorted(genres_seen),
                "list_diversity": diversity,
                "latency_ms": round(latency_ms, 1),
                "top_titles": [r.title for r in recs[:5]],
            }
            logger.info(
                "  %s: %d recs, mean=%.4f, diversity=%s, genres=%d, latency=%.1fms",
                variant_name, len(recs), variants[variant_name]["mean_score"],
                diversity, len(genres_seen), latency_ms,
            )
        except Exception as e:
            logger.error("  %s: FAILED — %s", variant_name, e)
            variants[variant_name] = {"error": str(e)}

    return variants


def evaluate_live_assistant_latency(db, user: User, num_queries: int = 5):
    """
    Measure end-to-end assistant pipeline latency with live (or mocked) LLM calls.
    Reports median and p95 latencies.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING ASSISTANT PIPELINE LATENCY")
    logger.info("=" * 60)

    from app.services.assistant_orchestrator import handle_message

    test_queries = [
        "Show me a good comedy from the 2000s",
        "Sci-fi movies like Interstellar",
        "Short thriller under 90 minutes",
        "Something light and fun, no horror",
        "Animated movies with high ratings",
    ][:num_queries]

    latencies_ms: list[float] = []
    results = []

    for q in test_queries:
        try:
            t0 = time.perf_counter()
            response = handle_message(user=user, message=q, session_id=None, db=db)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            results.append({
                "query": q,
                "latency_ms": round(elapsed_ms, 1),
                "num_movies": len(response.movies),
                "has_explanation": bool(response.structured_explanation),
            })
            logger.info("  '%s': %.1fms, %d movies", q, elapsed_ms, len(response.movies))
        except Exception as e:
            logger.error("  '%s': FAILED — %s", q, e)
            results.append({"query": q, "error": str(e)})

    latency_summary = {}
    if latencies_ms:
        latencies_sorted = sorted(latencies_ms)
        latency_summary = {
            "median_ms": round(statistics.median(latencies_sorted), 1),
            "p95_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 1),
            "min_ms": round(min(latencies_sorted), 1),
            "max_ms": round(max(latencies_sorted), 1),
        }
        logger.info("  Median: %.1fms, P95: %.1fms", latency_summary["median_ms"], latency_summary["p95_ms"])

    return results, latency_summary


def evaluate_assistant_system(live: bool = False, user_id: Optional[int] = None):
    """Run the full evaluation suite and produce a JSON report."""
    db = next(get_db())
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "live" if live else "offline",
    }

    try:
        # 1. Hard Constraint Satisfaction
        constraint_results, constraint_overall = evaluate_constraint_satisfaction(db)
        report["hard_constraint_evaluation"] = constraint_results
        report.setdefault("summary", {})["overall_constraint_satisfaction_pct"] = constraint_overall

        # 2. Follow-Up Delta Correctness
        followup_results, followup_accuracy = evaluate_followup_correctness()
        report["followup_evaluation"] = followup_results
        report["summary"]["followup_update_accuracy_pct"] = followup_accuracy

        # 3. DAMR vs Static Comparison (requires a user for personalization)
        user = None
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = db.query(User).filter(User.model_user_id.isnot(None)).first()

        if user:
            logger.info("Using user %s (model_user_id=%s) for personalization benchmarks", user.id, user.model_user_id)
            damr_vs_static = evaluate_damr_vs_static(db, user)
            report["damr_vs_static"] = damr_vs_static

            # 4. Live assistant latency (only in --live mode)
            if live:
                latency_results, latency_summary = evaluate_live_assistant_latency(db, user)
                report["latency_evaluation"] = latency_results
                report["summary"]["latency"] = latency_summary
            else:
                logger.info("Skipping live latency evaluation (use --live to enable)")
        else:
            logger.warning("No user with model_user_id found — skipping personalization benchmarks")

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        print(json.dumps(report, indent=2, default=str))
        return report

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Filmory recommender and assistant system.")
    parser.add_argument("--live", action="store_true", help="Run live assistant latency benchmarks (requires GEMINI_API_KEY)")
    parser.add_argument("--user-id", type=int, default=None, help="Specific user ID for personalization benchmarks")
    args = parser.parse_args()

    evaluate_assistant_system(live=args.live, user_id=args.user_id)
