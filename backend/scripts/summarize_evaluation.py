"""
Summarize Filmory academic evaluation suite (diagnostic run).

Reads:
  backend/evaluation/recommender_per_user.jsonl  (tune rows + test rows)
  backend/evaluation/assistant_per_query.jsonl
  backend/evaluation/split_manifest.json
  backend/evaluation/provenance.json

Statistics:
  * hit/recall rates -> Wilson score intervals (never a misleading zero-width
    interval: 0/200 reports [0.0000, 0.0184]);
  * NDCG means -> bootstrap percentile intervals;
  * ILD is range-checked against [0, 2] (1 - sim with sim in [-1, 1],
    diagonal excluded) and its similarity function is stated, not celebrated.

Only TEST-split recommender rows enter the reported numbers. Assistant metrics
use the narrowed label "evidence_reference_validity"; claim-level factual
support is reported as not evaluated. See evaluation/provenance.json: every
learned-model number here is DIAGNOSTIC (checkpoints trained on all
interactions), never a clean generalization claim.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BACKEND_DIR / "evaluation"

ILD_VALID_RANGE = (0.0, 2.0)
ILD_SIMILARITY = (
    "sim(i,j) = 0.5*cos(NCF-hybrid item emb) + 0.5*cos(genre vectors), "
    "ILD = mean pairwise (1 - sim), diagonal excluded, top-10 lists"
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def wilson_ci(hits: List[float], z: float = 1.96) -> Dict[str, float]:
    """Wilson score interval for a hit/recall rate."""
    n = len(hits)
    if n == 0:
        return {"mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0}
    p = float(sum(1.0 for h in hits if h) / n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"mean": round(p, 4), "ci95_lo": round(max(0.0, center - half), 4),
            "ci95_hi": round(min(1.0, center + half), 4), "n": n}


def bootstrap_mean_ci(values: List[float], n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    return {"mean": round(float(np.mean(arr)), 4),
            "ci95_lo": round(float(np.percentile(means, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(means, 97.5)), 4), "n": int(arr.size)}


def summarize_recommender(rows: List[Dict[str, Any]], catalog_items: int,
                           n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    test_rows = [r for r in rows if r.get("split", "test") == "test" and "systems" in r]
    systems = sorted({s for r in test_rows for s in r["systems"].keys()})
    track_b: Dict[str, Any] = {}
    ild_flags: List[str] = []
    for s in systems:
        hits = [1.0 if r["systems"][s].get("hit") else 0.0 for r in test_rows]
        ndcgs = [float(r["systems"][s].get("ndcg", 0.0)) for r in test_rows]
        ilds = [float(v) for r in test_rows
                for v in [r["systems"][s].get("ild")] if v is not None]
        bad = [v for v in ilds if not (ILD_VALID_RANGE[0] <= v <= ILD_VALID_RANGE[1])]
        if bad:
            ild_flags.append(f"{s}: {len(bad)} out-of-range ILD values")
        union: set = set()
        for r in test_rows:
            union.update(r["systems"][s].get("top10", []))
        track_b[s] = {
            "HR@10": wilson_ci(hits),
            "NDCG@10": bootstrap_mean_ci(ndcgs, n_boot, seed),
            "ILD@10_mean": round(float(np.mean(ilds)), 4) if ilds else None,
            "ILD_similarity": ILD_SIMILARITY,
            "catalog_coverage_pct": round(len(union) / max(1, catalog_items) * 100, 2),
            "users": len(hits),
        }

    retr_names = ("popularity", "baseline", "hybrid", "mixture")
    retrieval = {}
    for name in retr_names:
        vals = [1.0 if r.get("retrieval", {}).get(name, {}).get("recall100") else 0.0
                for r in test_rows]
        retrieval[name] = wilson_ci(vals)
    e2e_hits = [1.0 if r.get("track_a_e2e", {}).get("hit_10") else 0.0 for r in test_rows]
    e2e_ndcg = [float(r.get("track_a_e2e", {}).get("ndcg_10", 0.0)) for r in test_rows]
    e2e_rec = [1.0 if r.get("track_a_e2e", {}).get("candidate_recall_100") else 0.0
               for r in test_rows]
    e2e_lat = [float(r["track_a_e2e"]["latency_ms"]) for r in test_rows
               if r.get("track_a_e2e", {}).get("latency_ms") is not None]
    state_variants = sorted({r.get("state_variant") for r in test_rows})
    quality = sorted({r.get("quality_prior") for r in test_rows})
    track_a = {
        "retrieval_recall@100": retrieval,
        "end_to_end": {
            "candidate_recall_100": wilson_ci(e2e_rec),
            "HR@10": wilson_ci(e2e_hits),
            "NDCG@10": bootstrap_mean_ci(e2e_ndcg, n_boot, seed),
            "latency_p50_ms": round(float(np.percentile(e2e_lat, 50)), 2) if e2e_lat else None,
            "latency_p95_ms": round(float(np.percentile(e2e_lat, 95)), 2) if e2e_lat else None,
        },
        "state_variant": state_variants,
        "quality_prior": quality,
        "users": len(test_rows),
    }
    return {"track_a": track_a, "track_b_ablation": track_b, "ild_range_flags": ild_flags}


def summarize_assistant(rows: List[Dict[str, Any]], n_boot: int = 1000,
                        seed: int = 42) -> Dict[str, Any]:
    def _rate(key: str) -> Dict[str, float]:
        return wilson_ci([1.0 if r.get(key) else 0.0 for r in rows])

    rec = [r["resolution_recall"] for r in rows if r.get("resolution_recall") is not None]
    comp = [float(r.get("constraint_compliance", 0.0)) for r in rows]
    ext = [r["constraint_extraction_recall"] for r in rows
           if r.get("constraint_extraction_recall") is not None]
    llm_ms = [float(r["llm_ms"]) for r in rows if r.get("llm_ms")]
    src_counts: Dict[str, int] = {}
    for r in rows:
        s = r.get("extraction_source", "unknown")
        src_counts[s] = src_counts.get(s, 0) + 1
    models = sorted({str(r["model"]) for r in rows if r.get("model")})
    lat = [float(r.get("latency_ms", 0.0)) for r in rows]
    fup_rows = [r for r in rows if r.get("followup_pass") is not None]
    modes = sorted({r.get("assistant_mode", "offline-fallback") for r in rows})
    return {
        "queries": len(rows),
        "assistant_modes": modes,
        "models": models,
        "extraction_sources": src_counts,
        "routing_accuracy": _rate("routing_correct"),
        "constraint_extraction_recall_mean": round(float(np.mean(ext)), 4) if ext else None,
        "resolution_recall_mean": round(float(np.mean(rec)), 4) if rec else None,
        "constraint_compliance_mean": round(float(np.mean(comp)), 4) if comp else None,
        "evidence_reference_validity": _rate("evidence_reference_valid"),
        "claim_factual_support": "not evaluated (requires manual claim annotation)",
        "clarification_accuracy": _rate("clarification_correct"),
        "honest_empty_accuracy": _rate("empty_correct"),
        "followup_accuracy": wilson_ci(
            [1.0 if r.get("followup_pass") else 0.0 for r in fup_rows]) if fup_rows else None,
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2) if lat else None,
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
        "llm_ms_p50": round(float(np.percentile(llm_ms, 50)), 1) if llm_ms else None,
        "llm_ms_p95": round(float(np.percentile(llm_ms, 95)), 1) if llm_ms else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Filmory evaluation (diagnostic)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads((EVAL_DIR / "split_manifest.json").read_text(encoding="utf-8")) \
        if (EVAL_DIR / "split_manifest.json").exists() else {"metadata": {}}
    prov = json.loads((EVAL_DIR / "provenance.json").read_text(encoding="utf-8")) \
        if (EVAL_DIR / "provenance.json").exists() else {}
    catalog_items = int(manifest.get("metadata", {}).get("total_catalog_items", 0))

    rec_rows = _read_jsonl(EVAL_DIR / "recommender_per_user.jsonl")
    asst_dev = _read_jsonl(EVAL_DIR / "assistant_per_query.jsonl")
    asst_hidden = _read_jsonl(EVAL_DIR / "assistant_per_query_hidden.jsonl")
    asst_hidden_live = _read_jsonl(EVAL_DIR / "assistant_per_query_hidden_live.jsonl")
    rec_summary = summarize_recommender(rec_rows, catalog_items, args.n_boot, args.seed) if rec_rows else {}
    asst_summary = {
        "dev": summarize_assistant(asst_dev, args.n_boot, args.seed) if asst_dev else {},
        "hidden": summarize_assistant(asst_hidden, args.n_boot, args.seed) if asst_hidden else {},
        "hidden_live": summarize_assistant(asst_hidden_live, args.n_boot, args.seed) if asst_hidden_live else {},
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": ("DIAGNOSTIC — learned-model numbers are optimistic "
                   "(checkpoints trained on all interactions); see provenance.json"),
        "split": manifest.get("metadata", {}),
        "provenance": {"recorded": bool(prov),
                       "leakage_status": prov.get("leakage_status", "unknown")},
        "recommender": rec_summary,
        "assistant": asst_summary,
    }
    out = EVAL_DIR / "evaluation_summary.json"
    out.write_text(json.dumps(payload, indent=2))

    print("=" * 72)
    print("FILMORY EVALUATION SUMMARY (DIAGNOSTIC — see provenance.json)")
    print("=" * 72)
    if rec_summary:
        a = rec_summary["track_a"]
        print("\n--- Track A retrieval Recall@100 (Wilson 95%) ---")
        for name, m in a["retrieval_recall@100"].items():
            print(f"  {name:<12} {m['mean']:.4f} [{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}] (n={m['n']})")
        e = a["end_to_end"]
        print(f"End-to-end HR@10: {e['HR@10']['mean']:.4f} "
              f"[{e['HR@10']['ci95_lo']:.4f}, {e['HR@10']['ci95_hi']:.4f}] (n={e['HR@10']['n']})")
        print(f"state={a['state_variant']} quality={a['quality_prior']}")
        print("\n--- Track B (fixed 1+99; quality prior DISABLED in all DAMR rungs) ---")
        print(f"{'System':<28}{'HR@10':>10}{'NDCG@10':>10}{'ILD@10':>9}{'Coverage%':>11}")
        for name, m in rec_summary["track_b_ablation"].items():
            ild = m["ILD@10_mean"]
            print(f"{name:<28}{m['HR@10']['mean']:>10.4f}{m['NDCG@10']['mean']:>10.4f}"
                  f"{(ild if ild is not None else float('nan')):>9.4f}"
                  f"{m['catalog_coverage_pct']:>11.2f}")
        if rec_summary.get("ild_range_flags"):
            print("ILD RANGE FLAGS:", rec_summary["ild_range_flags"])
    if asst_summary.get("dev") or asst_summary.get("hidden"):
        print("\n--- Assistant (evidence-reference validity, NOT claim support) ---")
        for split_name in ("dev", "hidden"):
            part = asst_summary.get(split_name) or {}
            if not part:
                continue
            print(f"  [{split_name}]")
            for k, v in part.items():
                print(f"    {k}: {v}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
