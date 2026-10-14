"""Audit retrieval/ranking with frozen models; select on validation, test once.

No checkpoint, serving policy, database or research-paper mutations. The output
directory must be new. DAMR comparisons use positional state and disable quality
because this bundle contains no train-only per-movie ratings.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.config import settings
from app.ml.model_service import model_service as service
from app.ml.ranking_policy import select_candidate_positions, rank_candidate_positions, normalize_scores
from app.ml.damr import damr_rerank, estimate_user_state_from_positions, intra_list_diversity
from evaluate_candidate_policies import checkpoint_hashes


def key(config):
    return json.dumps(config, sort_keys=True)


def metric(rank, retrieved):
    return {'rank': rank, 'retrieved': bool(retrieved),
            'hr10': int(rank > 0), 'ndcg10': 1/math.log2(rank+1) if rank else 0.0}


def summarize(rows):
    size = len(rows)
    hits = sum(row['hr10'] for row in rows)
    retrieved = sum(row['retrieved'] for row in rows)
    return {'users': size, 'hr10': hits/size, 'ndcg10': sum(row['ndcg10'] for row in rows)/size,
            'candidate_recall': retrieved/size,
            'retrieval_misses': size-retrieved, 'retrieved_but_not_top10': retrieved-hits,
            'hits': hits, 'conditional_hr10_given_retrieved': hits/retrieved if retrieved else None}


def grid(mapped):
    weights = [[a/10, b/10, (10-a-b)/10] for a in range(11) for b in range(11-a)]
    weights.append([.55, .25, .20] if mapped else [.65, .25, .10])
    return [dict(candidate_k=k, strategy=strategy, weights=w)
            for k in [100, 300, 1000] for strategy in ['transformer', 'collaborative', 'mixed']
            for w in weights]


def baseline(mapped, length, policy):
    if mapped:
        return dict(candidate_k=100, strategy='transformer', weights=[.55,.25,.20])
    entry = next(entry for entry in policy['history_bins'] if entry['min_history'] == length)
    return {name:entry[name] for name in ['candidate_k','strategy','weights']}


@torch.inference_mode()
def signals(uid, length):
    history = [int(item) for item in service.user_sequences[uid]]
    context = history if length is None else history[-length:]
    mask = torch.ones(len(service.movie2idx), dtype=torch.bool, device=service.device)
    mask[context] = False
    items = torch.arange(len(mask), device=service.device)[mask]
    profile = service.user_genre_matrix[uid] if length is None else service.movie_genre_matrix[context].sum(0)
    if length is not None:
        profile = profile/profile.sum().clamp_min(1e-8)
    tr = service.sequential_transformer.score_candidates_with_sequence(context, items, service.device)
    genre = service.movie_genre_matrix[items] @ profile
    if length is None:
        expert = service.ncf_hybrid.score_candidate_items(uid, items, profile, service.movie_genre_matrix, service.device)
    else:
        from app.ml.ranking_policy import score_neighbor_items
        expert = score_neighbor_items(context, profile, service)[items]
    return items, expert, tr, genre, context


def score_policy(config, items, expert, tr, genre, target, pool=None):
    positions = pool if pool is not None else select_candidate_positions(tr, expert, genre, config['candidate_k'], config['strategy'])
    ranking, _ = rank_candidate_positions(positions, tr, expert, genre, config['weights'], 10)
    top = items[ranking].tolist()
    return metric(top.index(target)+1 if target in top else 0, bool((items[positions] == target).any())), top


def prior_cohorts():
    excluded = set()
    sources = []
    for path in list((BACKEND/'training/runs').glob('*/split.json')) + list((BACKEND/'evaluation').rglob('*.json')):
        if path.name not in {'split.json','metrics.json','selection.json','protocol.json'}:
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        found = False
        for field in ['validation_users','test_users','new_test_users','confirmation_users']:
            values = record.get(field, [])
            if values:
                excluded.update(values)
                found = True
        if found:
            sources.append(str(path.relative_to(BACKEND)))
    return excluded, sources


def paired(rows, chosen, seed):
    delta_hr = np.array([row[chosen]['hr10']-row['current']['hr10'] for row in rows])
    delta_ndcg = np.array([row[chosen]['ndcg10']-row['current']['ndcg10'] for row in rows])
    indices = np.random.default_rng(seed).integers(0, len(rows), size=(10000,len(rows)))
    result = {'users':len(rows), 'resamples':10000, 'seed':seed}
    for name, values in [('hr10',delta_hr), ('ndcg10',delta_ndcg)]:
        bounds = np.quantile(values[indices].mean(1), [.025,.975])
        result[name] = {'mean_difference':float(values.mean()), '95_percent_interval':bounds.tolist()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-users', type=int, default=400)
    parser.add_argument('--test-users', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026101401)
    args = parser.parse_args()
    if min(args.validation_users,args.test_users) < 1:
        raise ValueError('Evaluation cohorts must be nonempty')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    service.load_all()
    bundle = Path(service.ml_dir)
    split = json.loads((bundle/'split.json').read_text())
    policy_path = BACKEND/'ml/serving_policy.json'
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    hashes = checkpoint_hashes(bundle)
    if hashes != policy['checkpoint_hashes']:
        raise ValueError('Policy/checkpoint mismatch')
    val_users = [uid for uid in split['validation_users'] if len(service.user_sequences[uid]) >= 20][:args.validation_users]
    if len(val_users) != args.validation_users:
        raise ValueError('Requested validation cohort exceeds the stored split')
    excluded, sources = prior_cohorts()
    available = sorted(uid for uid in set(service.user_sequences)-excluded if len(service.user_sequences[uid]) >= 20)
    test_users = np.random.default_rng(args.seed).choice(available,args.test_users,replace=False).tolist()
    protocol = {'protocol':'Frozen clean checkpoints; identical full mapped catalogue; validation-only selection; fresh held-out test cohort',
        'validation_users':val_users,'new_test_users':test_users,'seed':args.seed,'checkpoint_hashes':hashes,
        'excluded_prior_users':len(excluded),'exclusion_sources':sources,
        'promotion_rule':'Selected policy must improve HR@10 and NDCG@10; both paired 95% lower bounds must exceed zero. No test-based reselection.',
        'modes':['mapped','live_2','live_5','live_10','live_20'],
        'minimum_train_history':20,
        'damr':'Secondary ablations on identical 100 Transformer candidates; quality disabled; positional state; freshness zero',
        'weights_grid':'0.1 simplex grid plus current mapped weights; candidate budgets 100/300/1000; Transformer/collaborative/mixed retrieval',
        'settings':{name:getattr(settings,name) for name in ['NCF_WEIGHT','TRANSFORMER_WEIGHT','GENRE_WEIGHT','DAMR_SLOPES','DAMR_ANCHOR','DAMR_ETA','DAMR_GAMMA','MMR_LAMBDA']}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for path in [Path(__file__),BACKEND/'app/ml/ranking_policy.py',BACKEND/'app/ml/damr.py']:
        shutil.copyfile(path,args.output/(path.stem+'_snapshot.py'))
    decisions = {}
    for length in [None,2,5,10,20]:
        mode = 'mapped' if length is None else 'live_'+str(length)
        configs = grid(length is None)
        current = baseline(length is None,length,policy)
        if current not in configs:
            configs.append(current)
        records = {key(config):[] for config in configs}
        groups = {}
        for config in configs:
            groups.setdefault((config['candidate_k'],config['strategy']),[]).append(config)
        with (args.output/(mode+'-validation.jsonl')).open('w',encoding='utf-8') as out:
            for number, uid in enumerate(val_users):
                items, expert, tr, genre, _ = signals(uid,length)
                target = split['validation_targets'][str(uid)]
                if not bool((items == target).any()):
                    raise ValueError('Held-out target excluded by history')
                row = {'user_idx':uid,'target_idx':target,'policies':{}}
                for pool_key, group in groups.items():
                    positions = select_candidate_positions(tr,expert,genre,*pool_key)
                    scores = torch.stack([normalize_scores(signal[positions]) for signal in [expert,tr,genre]],dim=0)
                    weights = torch.tensor([config['weights'] for config in group],device=service.device)
                    top = items[positions[(weights @ scores).topk(10,dim=1).indices]]
                    hits = top == target
                    ranks = torch.where(hits.any(1),hits.long().argmax(1)+1,0).tolist()
                    retrieved = bool((items[positions] == target).any())
                    for config, rank in zip(group,ranks):
                        result = metric(rank,retrieved)
                        records[key(config)].append(result)
                        row['policies'][key(config)] = {'rank':rank,'retrieved':retrieved}
                out.write(json.dumps(row)+'\n')
                if (number+1)%100 == 0:
                    print(f'{mode} validation {number+1}/{len(val_users)}',flush=True)
        summaries = {name:summarize(rows) for name,rows in records.items()}
        selected_key = max(summaries,key=lambda name:(summaries[name]['ndcg10'],summaries[name]['hr10'],name==key(current),-json.loads(name)['candidate_k']))
        decisions[mode] = {'current':current,'selected':json.loads(selected_key),'validation':summaries}
        print(f'{mode} selected {selected_key}',flush=True)
    # All five decisions are committed before any fresh test target is inspected.
    (args.output/'selection.json').write_text(json.dumps({**protocol,'decisions':decisions},indent=2),encoding='utf-8')
    report = {**protocol,'decisions':decisions,'test':{}}
    for length in [None,2,5,10,20]:
        mode = 'mapped' if length is None else 'live_'+str(length)
        results = []
        with (args.output/(mode+'-test.jsonl')).open('w',encoding='utf-8') as out:
            for number, uid in enumerate(test_users):
                items, expert, tr, genre, context = signals(uid,length)
                target = split['test_targets'][str(uid)]
                if not bool((items == target).any()):
                    raise ValueError('Held-out target excluded by history')
                row = {'user_idx':uid,'target_idx':target}
                configs = {'current':decisions[mode]['current'],'selected':decisions[mode]['selected'],
                    'Transformer':dict(candidate_k=100,strategy='transformer',weights=[0.,1.,0.]),
                    'Hybrid NCF' if length is None else 'Neighbor':dict(candidate_k=100,strategy='collaborative',weights=[1.,0.,0.])}
                for name,config in configs.items():
                    result, top = score_policy(config,items,expert,tr,genre,target)
                    row[name] = {**result,'top10':top}
                if length is None:
                    users_tensor = torch.full_like(items,uid)
                    base = service.ncf_baseline(users_tensor,items)
                    top = items[base.topk(10).indices].tolist()
                    pool = items[base.topk(100).indices]
                    row['NCF Baseline'] = {**metric(top.index(target)+1 if target in top else 0,bool((pool==target).any())),'top10':top}
                    positions = select_candidate_positions(tr,expert,genre,100,'transformer')
                    candidates = items[positions]
                    profile = estimate_user_state_from_positions(context)
                    for stage in range(5):
                        name = ['Fixed on DAMR pool','DAMR gate','DAMR + momentum','DAMR + agreement','DAMR + diversity'][stage]
                        ranked = damr_rerank(candidates,expert[positions],tr[positions],genre[positions],profile,
                            ratings=[None]*len(candidates),counts=[None]*len(candidates),top_k=10,
                            use_adaptive=stage>=1,use_momentum=stage>=2,use_agreement=stage>=3,
                            use_quality=False,use_diversity=stage>=4)
                        top = [entry['modelIdx'] for entry in ranked]
                        row[name] = {**metric(top.index(target)+1 if target in top else 0,bool((candidates==target).any())),
                            'top10':top,'ild10':intra_list_diversity(top)}
                results.append(row)
                out.write(json.dumps(row)+'\n')
                if (number+1)%100 == 0:
                    print(f'{mode} test {number+1}/{len(test_users)}',flush=True)
        names = [name for name in results[0] if name not in {'user_idx','target_idx'}]
        summaries = {name:summarize([row[name] for row in results]) for name in names}
        for name in names:
            if 'ild10' in results[0][name]:
                summaries[name]['ild10'] = sum(row[name]['ild10'] for row in results)/len(results)
        uncertainty = paired(results,'selected',args.seed+1)
        promote = (decisions[mode]['selected'] != decisions[mode]['current']
                   and all(uncertainty[name]['95_percent_interval'][0] > 0 for name in ['hr10','ndcg10']))
        report['test'][mode] = {'metrics':summaries,'paired_uncertainty':uncertainty,'promotion_supported':promote}
    report['limitations'] = ['Existing training users/items; live modes hide requester embedding and simulate short histories, not unseen people',
        'Single checkpoint seed; intervals cover user sampling only','Five exploratory policy comparisons; no prospective user satisfaction data',
        'DAMR positional state is a proxy, not measured temporal drift; Bayesian quality disabled to avoid unsupported metadata',
        'Classmate feedback must be collected separately; offline hit rates are not overall accuracy']
    assert checkpoint_hashes(bundle) == hashes and policy_path.read_bytes() == policy_bytes
    report['source_hashes'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in args.output.glob('*_snapshot.py')}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['test'],indent=2),flush=True)


if __name__ == '__main__':
    main()
