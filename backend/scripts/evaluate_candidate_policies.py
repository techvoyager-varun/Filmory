"""Select frozen-model serving policies on validation; evaluate once on new users.

Writes into a new output directory. Never edits checkpoints or the active bundle.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.ml.model_service import model_service
from app.ml.ranking_policy import select_candidate_positions, rank_candidate_positions


def checkpoint_hashes(bundle):
    return {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("ncf_baseline.pth", "ncf_hybrid.pth", "sequential_transformer.pth")}


def configurations(mapped):
    weights = [[.55, .25, .2], [.25, .65, .1], [0., 1., 0.]] if mapped else [[0., 1., 0.], [0., .9, .1], [0., .75, .25]]
    return [dict(candidate_k=k, strategy=strategy, weights=w)
            for k in (100, 300, 1000) for strategy in ("transformer", "mixed") for w in weights]


def key(config):
    return json.dumps(config, sort_keys=True)


@torch.inference_mode()
def evaluate(users, targets, configs, mapped, output):
    service = model_service
    records = []
    item_count = len(service.movie2idx)
    all_items = torch.arange(item_count, device=service.device)
    started = time.perf_counter()
    for index, uid in enumerate(users):
        history = service.user_sequences[uid]
        eligible = torch.ones(item_count, dtype=torch.bool, device=service.device)
        eligible[history] = False
        items = all_items[eligible]
        tr = service.sequential_transformer.score_candidates_with_sequence(history, items, service.device)
        profile = service.user_genre_matrix[uid]
        genre = service.movie_genre_matrix[items] @ profile
        if mapped:
            ncf = service.ncf_hybrid.score_candidate_items(uid, items, profile, service.movie_genre_matrix, service.device)
        else:
            # Item-only similarity is available for users without learned user embeddings.
            embeddings = service.ncf_hybrid.item_embedding.weight
            recent = embeddings[history[-service.sequential_transformer.max_len:]].mean(0)
            ncf = torch.nn.functional.cosine_similarity(embeddings[items], recent.unsqueeze(0), dim=-1)
        target = targets[str(uid)]
        row = {"user_idx":uid, "target_idx":target, "policies":{}}
        for config in configs:
            positions = select_candidate_positions(tr, ncf, genre, config["candidate_k"], config["strategy"])
            ranking, _ = rank_candidate_positions(positions, tr, ncf, genre, config["weights"], 10)
            top = items[ranking].tolist()
            row["policies"][key(config)] = {"rank":top.index(target)+1 if target in top else 0,
                "retrieved":bool((items[positions] == target).any()), "top10":top}
        records.append(row)
        if (index+1) % 50 == 0:
            print(f"{'mapped' if mapped else 'sequence-only'}: {index+1}/{len(users)} profiles", flush=True)
    output.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')
    summaries = {}
    for config in configs:
        rows = [row["policies"][key(config)] for row in records]
        ranks = [row['rank'] for row in rows]
        summaries[key(config)] = {"users":len(rows), "hr10":sum(r>0 for r in ranks)/len(rows),
            "ndcg10":sum(1/math.log2(r+1) for r in ranks if r)/len(rows),
            "candidate_recall":sum(row['retrieved'] for row in rows)/len(rows),
            "coverage10":len({item for row in rows for item in row['top10']})/item_count}
    return summaries, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model_service.load_all()
    bundle = Path(model_service.ml_dir)
    hashes = checkpoint_hashes(bundle)
    split = json.loads((bundle/'split.json').read_text())
    validation_users = split['validation_users'][:args.users]
    # Previous papers inspected both stored test subsets. Use additional users
    # whose held-out targets were not part of those reported evaluation cohorts.
    prior = set(split['validation_users']) | set(split['test_users'])
    available = sorted(set(model_service.user_sequences) - prior)
    new_test_users = np.random.default_rng(20261013).choice(available, args.users, replace=False).tolist()
    report = {"protocol":"Frozen checkpoints; full mapped catalogue; train history excluded; policy selected on validation NDCG@10",
        "bundle":str(bundle), "checkpoint_hashes":hashes, "seed":20261013,
        "validation_users":validation_users, "new_test_users":new_test_users, "modes":{}}
    for mode, mapped in [('mapped',True),('sequence_only',False)]:
        configs = configurations(mapped)
        validation, seconds = evaluate(validation_users, split['validation_targets'], configs, mapped, args.output/f'{mode}-validation.jsonl')
        selected_key = max(validation, key=lambda k:(validation[k]['ndcg10'], validation[k]['hr10'], -json.loads(k)['candidate_k'], json.loads(k)['strategy']=='transformer'))
        selected = json.loads(selected_key)
        baseline = dict(candidate_k=100,strategy='transformer',weights=[.55,.25,.2] if mapped else [0.,.75,.25])
        report['modes'][mode] = {"validation":validation, "selected":selected, "validation_seconds":seconds}
        # Selection is frozen before the new test targets are evaluated.
        (args.output/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        test_configs = [selected] if selected == baseline else [baseline,selected]
        test, seconds = evaluate(new_test_users, split['test_targets'], test_configs, mapped, args.output/f'{mode}-new-test.jsonl')
        report['modes'][mode].update(new_test=test,new_test_seconds=seconds,baseline=baseline)
    assert checkpoint_hashes(bundle) == hashes
    report['limitations'] = ['Single frozen training seed and bounded cohorts',
        'Sequence-only mode hides the user embedding but replays existing train histories; not a prospective real-user study',
        'No quality metadata or timestamp-based feedback in this offline experiment',
        'New test uses held-out targets of additional training users, not entirely unseen people']
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({mode:{'selected':data['selected'],'new_test':data['new_test']} for mode,data in report['modes'].items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
