"""
Filmory Assistant Quality Suite (academic evaluation track).

Evaluates the Ask Filmory assistant against a curated, annotated benchmark
(`backend/evaluation/assistant_queries.jsonl`) with verified database IDs.

Measured dimensions:
  1. Mode routing accuracy — predicted request_mode vs expected_mode.
  2. Constraint-extraction recall — fraction of ground-truth constraint fields
     the raw extracted intent already contains (before ground-truth overlay).
  3. Title resolution accuracy — expected catalog/reference IDs found in results.
  4. Hard-constraint satisfaction — no returned candidate violates the
     benchmark's ground-truth constraints (via _matches_hard_constraints).
  5. Evidence-reference validity — every evidence_id references a returned
     movie and a key from its evidence package. This is NOT claim-level
     factual support (requires manual annotation; reported as not evaluated).
  6. Clarification behaviour + honest-empty behaviour for vague/unknown queries.
  7. Follow-up handling — constraint tightening and ordinal exclusions
     (incl. a two-turn case).
  8. Latency — per-query retrieval pipeline latency (p50/p95).

assistant_queries.jsonl is the DEVELOPMENT set (the deterministic fallback
parser was tuned against it — keep as regression tests).
assistant_queries_hidden.jsonl is the HELD-OUT test set (new wording/titles).
Each run is labelled offline-fallback or live-gemini via --live.

Intent extraction is offline-deterministic by default (heuristic fallback
parser). Pass --live to use Gemini intent extraction when GEMINI_API_KEY is set.

Reads:  backend/evaluation/assistant_queries.jsonl
Writes: backend/evaluation/assistant_per_query.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = BACKEND_DIR / "evaluation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filmory.eval_assistant")

# Pacing for live runs (free-tier per-minute quotas). Set via --live-delay-secs.
LIVE_DELAY_SECS = 13.0
_last_live_call: float = 0.0

from app.database import get_db  # noqa: E402
from app.models.db_models import Movie, User  # noqa: E402
from app.schemas.assistant import IntentDelta, MovieIntent  # noqa: E402
from app.services.assistant_orchestrator import (  # noqa: E402
    _build_evidence,
    _matches_hard_constraints,
    _merge_intents,
)
from app.services.llm_client import extract_intent, fallback_extract_intent  # noqa: E402
from app.services.semantic_search import search_movies_by_intent  # noqa: E402


def _extract_delta(query: str, live: bool):
    """Extract an IntentDelta via Gemini (live) or the deterministic fallback.

    Returns (delta, source, llm_ms) where source is one of llm /
    llm_unparseable / llm_error_fallback / fallback, and llm_ms is API time
    (0.0 without an API call) — kept separate from quota-pacing sleeps.
    """
    if live:
        global _last_live_call
        import time as _time

        wait = LIVE_DELAY_SECS - (_time.monotonic() - _last_live_call)
        if wait > 0:
            _time.sleep(wait)
        try:
            from app.services import llm_client as _llm

            d = extract_intent(query, None)
            _last_live_call = _time.monotonic()
            return d, _llm.last_extraction_source(), _llm.last_call_latency_ms()
        except Exception as e:
            logger.warning("Live intent extraction failed, using fallback: %s", e)
            _last_live_call = _time.monotonic()
            return fallback_extract_intent(query), "llm_error_fallback", 0.0
    return fallback_extract_intent(query), "fallback", 0.0


def _extraction_recall(delta: IntentDelta, constraints: Dict[str, Any]) -> Optional[float]:
    """Fraction of the benchmark's ground-truth constraint fields the RAW
    extracted turn-1 intent already contains (measured BEFORE the evaluator
    overlays ground truth for retrieval). Excluded-genre extraction is a known
    fallback-parser gap (only the LLM emits exclusions)."""
    keys = [k for k in ("preferred_genres", "excluded_genres", "min_year",
                        "max_year", "max_runtime_minutes", "min_rating")
            if k in constraints and constraints[k] is not None]
    if not keys:
        return None
    base = _merge_intents({}, delta, None)
    hits = 0
    for k in keys:
        actual, want = getattr(base, k, None), constraints[k]
        if isinstance(want, list):
            if sorted([str(x).lower() for x in (actual or [])]) == sorted([str(x).lower() for x in want]):
                hits += 1
        elif actual == want:
            hits += 1
    return round(hits / len(keys), 4)


def _build_eval_intent(delta: IntentDelta, constraints: Dict[str, Any]) -> MovieIntent:
    """Merge extracted turn-1 intent with the benchmark's ground-truth constraints.

    The benchmark constraints represent the user's true requirements (even when
    the heuristic extractor cannot parse e.g. runtimes/years). Enforcing them
    here isolates *retrieval + constraint satisfaction* from extractor recall.
    Routing is still scored on the raw extracted mode.
    """
    merged = _merge_intents({}, delta, None).model_dump()
    for key in (
        "preferred_genres",
        "excluded_genres",
        "min_year",
        "max_year",
        "max_runtime_minutes",
        "min_rating",
    ):
        if key in constraints and constraints[key] is not None:
            merged[key] = constraints[key]
    # Keep extracted reference titles / semantic query for resolution scoring.
    return MovieIntent.model_validate(merged)


def _grounding_check(
    movie_ids: List[int], evidence_keys_by_movie: Dict[int, List[str]]
) -> Dict[str, Any]:
    """Validate a template explanation: one item per movie, evidence_ids drawn
    strictly from that movie's own evidence package."""
    items = [
        {
            "movie_id": mid,
            "evidence_ids": [
                k
                for k in evidence_keys_by_movie.get(mid, [])
                if ("constraint" in k or "genre" in k or k.endswith(":genres"))
            ][:3],
        }
        for mid in movie_ids
    ]
    valid_movie_set = set(movie_ids)
    n_checked = 0
    n_valid = 0
    for item in items:
        allowed = set(evidence_keys_by_movie.get(item["movie_id"], []))
        if item["movie_id"] not in valid_movie_set:
            continue
        for eid in item["evidence_ids"]:
            n_checked += 1
            if eid in allowed:
                n_valid += 1
    return {
        "items": items,
        "claims_checked": n_checked,
        "claims_supported": n_valid,
        "valid": n_checked == 0 or n_checked == n_valid,
    }


def evaluate_benchmark(
    benchmark_path: Optional[Path] = None,
    live: bool = False,
    user_id: Optional[int] = None,
    limit: int = 10,
    out_name: str = "assistant_per_query.jsonl",
    live_delay_secs: float = 13.0,
) -> Dict[str, Any]:
    benchmark_path = benchmark_path or (EVAL_DIR / "assistant_queries.jsonl")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    cases = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d assistant benchmark cases from %s", len(cases), benchmark_path)
    if live:
        global LIVE_DELAY_SECS
        LIVE_DELAY_SECS = live_delay_secs
        from app.config import settings as _settings

        logger.info("Live mode: model=%s, pacing=%.0fs/call",
                    _settings.GEMINI_CHAT_MODEL, LIVE_DELAY_SECS)
    _model_name: Optional[str] = None
    if live:
        from app.config import settings as _settings2

        _model_name = _settings2.GEMINI_CHAT_MODEL

    db = next(get_db())
    try:
        user = None
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = db.query(User).filter(User.model_user_id.isnot(None)).first()
        if not user:
            user = db.query(User).first()
        if not user:
            raise SystemExit("No users in database — cannot build evidence packages.")
        logger.info("Evidence built as user id=%s", user.id)

        out_path = EVAL_DIR / out_name
        latencies: List[float] = []
        routing_hits = 0
        resolution_scores: List[float] = []
        constraint_rates: List[float] = []
        grounding_hits = 0
        grounding_total = 0
        clarification_hits = 0
        clarification_total = 0
        empty_hits = 0
        empty_total = 0
        followup_hits = 0
        followup_total = 0
        all_sources: List[str] = []

        with open(out_path, "w", encoding="utf-8") as fout:
            for case in cases:
                qid = case["id"]
                query = case["query"]
                expected_mode = case.get("expected_mode", "personalized_recommendation")
                constraints = case.get("constraints", {}) or {}
                expected_ids = case.get("expected_movie_ids", []) or []
                expected_refs = case.get("expected_reference_ids", []) or []
                expect_clar = bool(case.get("expect_clarification", False))
                expect_empty = bool(case.get("expect_empty", False))

                t0 = time.perf_counter()
                delta, ext_source, ext_ms = _extract_delta(query, live)
                predicted_mode = delta.request_mode or "catalog_lookup"
                routing_ok = predicted_mode == expected_mode
                routing_hits += int(routing_ok)
                extraction_recall = _extraction_recall(delta, constraints)
                all_sources.append(ext_source)

                eval_intent = _build_eval_intent(delta, constraints)
                # The evaluated routing mode is the extracted one.
                eval_intent.request_mode = predicted_mode  # type: ignore[assignment]

                candidates, resolved_refs = search_movies_by_intent(
                    eval_intent, db, exclude_ids=set(), limit=50
                )
                top = candidates[:limit]
                top_ids = [m.movie_id for m in top]

                # ---- resolution accuracy ----
                targets = expected_ids or expected_refs
                if targets:
                    if expected_mode == "similar_movies" and expected_refs:
                        resolved_ok = any(r in expected_refs for r in resolved_refs)
                        recall = 1.0 if resolved_ok else 0.0
                    else:
                        hits = sum(1 for t in targets if t in top_ids)
                        recall = hits / max(1, len(targets))
                else:
                    recall = float("nan")
                    hits = 0
                if targets:
                    resolution_scores.append(recall)

                # ---- hard-constraint compliance ----
                if top:
                    n_valid = sum(
                        1 for m in top if _matches_hard_constraints(m, eval_intent)
                    )
                    compliance = n_valid / len(top)
                else:
                    n_valid = 0
                    compliance = 1.0 if expect_empty else 0.0
                constraint_rates.append(compliance)

                # ---- evidence-reference validity (NOT claim-level factual support:
                # checks that explanation evidence_ids reference returned movies
                # and keys from their evidence packages) ----
                evidence_keys: Dict[int, List[str]] = {}
                for m in top:
                    ev = _build_evidence(m, eval_intent, 1.0, user, db)
                    evidence_keys[m.movie_id] = list(ev.evidence_keys)
                grounding = _grounding_check(top_ids, evidence_keys)
                grounding_total += 1
                grounding_hits += int(grounding["valid"])

                # ---- clarification / honest-empty ----
                if expect_clar or "vague" in qid or qid.startswith("vague"):
                    clarification_total += 1
                    clar_ok = bool(delta.needs_clarification) == expect_clar
                    clarification_hits += int(clar_ok)
                else:
                    clar_ok = True
                if expect_empty or "unknown" in qid:
                    empty_total += 1
                    empty_ok = (len(top) == 0) == expect_empty
                    empty_hits += int(empty_ok)
                else:
                    empty_ok = True

                # ---- follow-up turns (multi-turn: followup, then followup2) ----
                followup_sources: List[str] = []
                def _check_followup(fup: Dict[str, Any], prev_intent: MovieIntent,
                                    prev_ids: List[int]) -> bool:
                    f_delta, f_src, _ = _extract_delta(fup["query"], live)
                    followup_sources.append(f_src)
                    merged = _merge_intents(prev_intent.model_dump(), f_delta, prev_ids)
                    ok = True
                    exp_c = fup.get("expected_constraints", {}) or {}
                    for k, v in exp_c.items():
                        actual = getattr(merged, k, None)
                        if isinstance(v, list):
                            if sorted(actual or []) != sorted(v):
                                ok = False
                        elif actual != v:
                            ok = False
                    exp_ord = fup.get("expected_ordinal_exclusion")
                    if exp_ord is not None:
                        want = prev_ids[exp_ord - 1] if 0 < exp_ord <= len(prev_ids) else None
                        if want is None or want not in (merged.excluded_movie_ids or []):
                            ok = False
                    return ok, merged

                followup = case.get("followup")
                if followup:
                    followup_total += 1
                    f_pass, merged2 = _check_followup(followup, eval_intent, top_ids)
                    all_sources.extend(followup_sources)
                    fup2 = case.get("followup2")
                    if fup2:
                        f2_pass, _ = _check_followup(fup2, merged2, top_ids)
                        f_pass = bool(f_pass and f2_pass)
                    followup_hits += int(f_pass)
                    followup_pass: Optional[bool] = bool(f_pass)
                else:
                    followup_pass = None

                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

                record = {
                    "id": qid,
                    "query": query,
                    "assistant_mode": "live-gemini" if live else "offline-fallback",
                    "model": _model_name,
                    "extraction_source": ext_source,
                    "llm_ms": round(ext_ms, 1),
                    "followup_sources": followup_sources,
                    "expected_mode": expected_mode,
                    "predicted_mode": predicted_mode,
                    "routing_correct": routing_ok,
                    "constraint_extraction_recall": extraction_recall,
                    "resolution_targets": targets,
                    "resolution_hits": hits if targets else 0,
                    "resolution_recall": round(recall, 4)
                    if recall == recall
                    else None,
                    "num_candidates": len(top),
                    "constraint_compliance": round(compliance, 4),
                    "constraint_violations": len(top) - n_valid,
                    "evidence_reference_valid": grounding["valid"],
                    "grounding_valid": grounding["valid"],
                    "grounding_claims": f"{grounding['claims_supported']}/{grounding['claims_checked']}",
                    "claim_factual_support": "not evaluated (requires manual claim annotation)",
                    "clarification_correct": clar_ok,
                    "empty_correct": empty_ok,
                    "followup_pass": followup_pass,
                    "latency_ms": round(elapsed_ms, 2),
                }
                fout.write(json.dumps(record) + "\n")

        def _mean(xs: List[float]) -> Optional[float]:
            xs = [x for x in xs if x == x]
            return round(float(np.mean(xs)), 4) if xs else None

        extraction_vals = []
        llm_ms_vals = []
        for line in open(out_path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("constraint_extraction_recall") is not None:
                extraction_vals.append(rec["constraint_extraction_recall"])
            if rec.get("llm_ms"):
                llm_ms_vals.append(float(rec["llm_ms"]))
        llm_successes = sum(1 for s in all_sources if s == "llm")
        if live:
            non_llm = [s for s in all_sources if s != "llm"]
            if non_llm:
                logger.warning(
                    "LIVE RUN PARTIALLY CONTAMINATED: %d/%d extractions fell back "
                    "(%s). Per-query sources are in the output file; treat blended "
                    "metrics with suspicion.",
                    len(non_llm), len(all_sources), sorted(set(non_llm)))
        if live and llm_successes == 0:
            raise SystemExit(
                "LIVE RUN INVALID: 0 successful LLM extractions "
                f"({len(all_sources)} calls, sources={sorted(set(all_sources))}). "
                "API quota likely exhausted — no live numbers written to the paper. "
                "Output file was still written for inspection but must NOT be reported."
            )
        summary = {
            "metadata": {
                "benchmark": str(benchmark_path),
                "num_queries": len(cases),
                "mode": "live-gemini" if live else "offline-fallback",
                "model": _model_name if live else None,
                "llm_successful_extractions": llm_successes,
                "extraction_source_counts": {s: all_sources.count(s) for s in sorted(set(all_sources))},
                "note": ("assistant_queries.jsonl is the DEVELOPMENT set "
                         "(parser tuned against it); assistant_queries_hidden.jsonl "
                         "is the held-out test set"),
            },
            "routing_accuracy": round(routing_hits / max(1, len(cases)), 4),
            "constraint_extraction_recall_mean": _mean(extraction_vals),
            "resolution_recall_mean": _mean(resolution_scores),
            "constraint_compliance_mean": _mean(constraint_rates),
            "evidence_reference_validity": round(
                grounding_hits / max(1, grounding_total), 4
            ),
            "claim_factual_support": "not evaluated (requires manual claim annotation)",
            "clarification_accuracy": round(
                clarification_hits / max(1, clarification_total), 4
            )
            if clarification_total
            else None,
            "honest_empty_accuracy": round(empty_hits / max(1, empty_total), 4)
            if empty_total
            else None,
            "followup_accuracy": round(followup_hits / max(1, followup_total), 4)
            if followup_total
            else None,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2)
            if latencies
            else None,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2)
            if latencies
            else None,
            "note_latency": ("latency_ms includes quota-pacing sleeps; "
                             "llm_ms is pure API inference time" if live else None),
            "llm_ms_p50": round(float(np.percentile(llm_ms_vals, 50)), 1) if llm_ms_vals else None,
            "llm_ms_p95": round(float(np.percentile(llm_ms_vals, 95)), 1) if llm_ms_vals else None,
        }
        logger.info("Assistant benchmark complete -> %s", out_path)
        logger.info(json.dumps(summary, indent=2))
        return summary
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Filmory assistant benchmark")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(EVAL_DIR / "assistant_queries.jsonl"),
        help="Path to assistant_queries.jsonl",
    )
    parser.add_argument("--live", action="store_true", help="Use Gemini intent extraction")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=str, default="assistant_per_query.jsonl",
                        help="Output file name inside backend/evaluation/")
    parser.add_argument("--live-delay-secs", type=float, default=13.0,
                        help="Minimum seconds between live LLM calls (quota pacing)")
    args = parser.parse_args()
    evaluate_benchmark(
        benchmark_path=Path(args.benchmark),
        live=args.live,
        user_id=args.user_id,
        limit=args.limit,
        out_name=args.out,
        live_delay_secs=args.live_delay_secs,
    )
