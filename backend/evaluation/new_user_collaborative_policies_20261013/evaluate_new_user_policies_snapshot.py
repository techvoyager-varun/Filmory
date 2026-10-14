"""Validate cold-start versus sequence ranking at realistic short history lengths."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
from app.ml.model_service import model_service
from app.ml.ranking_policy import select_candidate_positions, rank_candidate_positions, score_neighbor_items
from evaluate_candidate_policies import checkpoint_hashes, configurations, key


@torch.inference_mode()
def legacy_ranking(neighbor_scores, eligible):
    scores=neighbor_scores.clone()
    scores[~eligible]=-torch.inf
    return scores.topk(min(10,int(eligible.sum()))).indices.tolist()


@torch.inference_mode()
def evaluate(users,targets,context_len,configs,path):
    s=model_service
    all_items=torch.arange(len(s.movie2idx),device=s.device)
    records=[]
    for index,uid in enumerate(users):
        context=s.user_sequences[uid][-context_len:]
        eligible=torch.ones(len(all_items),dtype=torch.bool,device=s.device)
        eligible[context]=False
        items=all_items[eligible]
        profile=s.movie_genre_matrix[context].sum(0)
        profile=profile/profile.sum().clamp_min(1e-8)
        tr=s.sequential_transformer.score_candidates_with_sequence(context,items,s.device)
        genre=s.movie_genre_matrix[items]@profile
        neighbor_scores=score_neighbor_items(context,profile,s)
        collaborative=neighbor_scores[items]
        top10={'legacy_cold_start':legacy_ranking(neighbor_scores,eligible)}
        candidate_recall={}
        target=targets[str(uid)]
        for config in configs:
            positions=select_candidate_positions(tr,collaborative,genre,config['candidate_k'],config['strategy'])
            ranking,_=rank_candidate_positions(positions,tr,collaborative,genre,config['weights'],10)
            top10[key(config)]=items[ranking].tolist()
            candidate_recall[key(config)]=bool((items[positions]==target).any())
        records.append({'user_idx':uid,'context_len':len(context),'target_idx':target,'rank':{
            name:top.index(target)+1 if target in top else 0 for name,top in top10.items()},'top10':top10,'retrieved':candidate_recall})
        if (index+1)%100==0: print(f"History {context_len}: {index+1}/{len(users)}",flush=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    result={}
    for name in records[0]['rank']:
        ranks=[row['rank'][name] for row in records]
        result[name]={'users':len(records),'hr10':sum(r>0 for r in ranks)/len(ranks),
            'ndcg10':sum(1/math.log2(r+1) for r in ranks if r)/len(ranks),
            'coverage10':len({item for row in records for item in row['top10'][name]})/len(all_items)}
        if name!='legacy_cold_start': result[name]['candidate_recall']=sum(row['retrieved'][name] for row in records)/len(records)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-evaluation',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261015)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model_service.load_all()
    bundle=Path(model_service.ml_dir)
    split=json.loads((bundle/'split.json').read_text())
    prior=json.loads((args.prior_evaluation/'metrics.json').read_text())
    excluded=set(split['validation_users'])|set(split['test_users'])|set(prior['new_test_users'])
    available=sorted(set(model_service.user_sequences)-excluded)
    # Also exclude the initial short-history investigation's test cohort.
    initial=BACKEND/'evaluation/new_user_policies_20261013/selection.json'
    if initial.exists():
        excluded.update(json.loads(initial.read_text())['new_test_users'])
        available=sorted(set(model_service.user_sequences)-excluded)
    test_users=np.random.default_rng(args.seed).choice(available,200,replace=False).tolist()
    validation_users=split['validation_users'][:200]
    configs=[dict(candidate_k=k,strategy=strategy,weights=weights)
        for k in (100,300,1000) for strategy in ('transformer','collaborative','mixed')
        for weights in ([0.,.75,.25],[.4,.5,.1],[.65,.25,.1],[.8,.2,0.])]
    report={'checkpoint_hashes':checkpoint_hashes(bundle),'seed':args.seed,'validation_users':validation_users,
        'new_test_users':test_users,'history_lengths':{},'selection_metric':'validation NDCG@10',
        'protocol':'Hide learned user ID; use only the last 2/5/10/20 train events and their genres; exclude only those visible events; full mapped catalogue; held-out targets not used for fitting'}
    # Freeze every history-bin decision before inspecting any new test results.
    for length in (2,5,10,20):
        metrics=evaluate(validation_users,split['validation_targets'],length,configs,args.output/f'validation-{length}.jsonl')
        selected=max(metrics,key=lambda name:(metrics[name]['ndcg10'],metrics[name]['hr10'],name=='legacy_cold_start',-json.loads(name)['candidate_k'] if name!='legacy_cold_start' else 0))
        policy={'mode':'legacy_cold_start'} if selected=='legacy_cold_start' else {'mode':'sequence_only',**json.loads(selected)}
        report['history_lengths'][str(length)]={'validation':metrics,'selected':policy}
        print('Selected history',length,policy,flush=True)
    (args.output/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for length in (2,5,10,20):
        policy=report['history_lengths'][str(length)]['selected']
        chosen=[{k:v for k,v in policy.items() if k!='mode'}] if policy['mode']=='sequence_only' else []
        metrics=evaluate(test_users,split['test_targets'],length,chosen,args.output/f'new-test-{length}.jsonl')
        report['history_lengths'][str(length)]['new_test']=metrics
    report['limitations']=['History truncation simulates a new account; users/items were present in the training population',
        'One frozen training seed; 200 validation users and 200 additional test users',
        'No user-satisfaction experiment or timestamps',
        'Choices across history bins are exploratory; confidence intervals and prospective users are needed']
    assert checkpoint_hashes(bundle)==report['checkpoint_hashes']
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({length:{'selected':m['selected'],'new_test':m['new_test']} for length,m in report['history_lengths'].items()},indent=2),flush=True)


if __name__=='__main__':main()
