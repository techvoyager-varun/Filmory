"""Confirm the already selected local serving policy on additional users.

No fitting, hyperparameter selection or policy changes occur in this script.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.ml.model_service import model_service
from evaluate_candidate_policies import checkpoint_hashes, key
from evaluate_new_user_policies import evaluate


def bootstrap_differences(directory, users, bins, seed, resamples=10000):
    differences = {"hr10": [], "ndcg10": []}
    for length, policy in bins.items():
        name = key({k:v for k,v in policy.items() if k not in {"mode", "min_history"}})
        rows = [json.loads(line) for line in (directory/f"confirmation-{length}.jsonl").read_text().splitlines()]
        assert [row["user_idx"] for row in rows] == users
        differences["hr10"].append(np.array([
            int(row["rank"][name] > 0) - int(row["rank"]["legacy_cold_start"] > 0) for row in rows]))
        differences["ndcg10"].append(np.array([
            (1/np.log2(row["rank"][name]+1) if row["rank"][name] else 0) -
            (1/np.log2(row["rank"]["legacy_cold_start"]+1) if row["rank"]["legacy_cold_start"] else 0) for row in rows]))
    result = {"protocol":"Paired user-level percentile bootstrap; average four history contexts per user before resampling",
              "users":len(users), "resamples":resamples, "seed":seed}
    indices = np.random.default_rng(seed).integers(0, len(users), size=(resamples, len(users)))
    for metric, rows in differences.items():
        per_user = np.stack(rows).mean(0)
        low, high = np.quantile(per_user[indices].mean(1), [.025, .975])
        result[metric] = {"mean_difference":float(per_user.mean()), "95_percent_interval":[float(low),float(high)]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.users < 1:
        raise ValueError("At least one evaluation user is required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model_service.load_all()
    bundle = Path(model_service.ml_dir)
    split = json.loads((bundle/"split.json").read_text())
    policy = json.loads((BACKEND/"ml/serving_policy.json").read_text())
    hashes = checkpoint_hashes(bundle)
    assert policy["checkpoint_hashes"] == hashes
    excluded = set()
    exclusion_files = list((BACKEND/"training/runs").glob("*/split.json"))
    exclusion_files += list((BACKEND/"evaluation").glob("*/metrics.json"))
    exclusion_files += list((BACKEND/"evaluation").glob("*/selection.json"))
    for path in exclusion_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in ("validation_users", "test_users", "new_test_users", "confirmation_users"):
            excluded.update(record.get(field, []))
    available = sorted(set(model_service.user_sequences) - excluded)
    if len(available) < args.users:
        raise ValueError("Not enough additional users outside recorded earlier cohorts")
    users = np.random.default_rng(args.seed).choice(available, args.users, replace=False).tolist()
    bins = {str(entry["min_history"]):entry for entry in policy["history_bins"]}
    if any(entry["mode"] != "sequence_only" for entry in bins.values()):
        raise ValueError("This comparison requires a selected sequence policy in every history bin")
    record = {"protocol":"Confirm the frozen serving policy without reselection; hide requester embedding; short train-history contexts; full mapped catalogue",
              "seed":args.seed, "confirmation_users":users, "excluded_prior_users":len(excluded),
              "exclusion_sources":[str(path.relative_to(BACKEND)) for path in exclusion_files],
              "checkpoint_hashes":hashes, "frozen_policy":policy, "history_lengths":{}}
    # Save decisions/cohort before calculating any confirmation outcome.
    (args.output/"protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for source in [Path(__file__), BACKEND/"scripts/evaluate_new_user_policies.py", BACKEND/"app/ml/ranking_policy.py"]:
        shutil.copyfile(source, args.output/(source.stem+"_snapshot.py"))
    for length, entry in bins.items():
        config = {k:v for k,v in entry.items() if k not in {"mode", "min_history"}}
        metrics = evaluate(users, split["test_targets"], int(length), [config], args.output/f"confirmation-{length}.jsonl")
        record["history_lengths"][length] = {"policy":entry, "metrics":metrics}
    record["paired_uncertainty"] = bootstrap_differences(args.output, users, bins, args.seed+1)
    record["limitations"] = ["Users/items existed during training; this is not an unseen-person or prospective user study",
        "One frozen checkpoint seed; bootstrap covers user sampling, not training variability",
        "Policy and earlier validation cohort remain unchanged; this confirmation is reported whether positive or negative"]
    assert checkpoint_hashes(bundle) == hashes
    assert json.loads((BACKEND/"ml/serving_policy.json").read_text()) == policy
    record["source_hashes"] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in args.output.glob("*_snapshot.py")}
    (args.output/"metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"history_lengths":record["history_lengths"],"paired_uncertainty":record["paired_uncertainty"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
