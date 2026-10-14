"""Recompute manuscript metrics and check their source-artifact lineage."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
from app.ml.model_service import model_service
from train_models import evaluate


def wilson(hits,users):
    z=1.959963984540054
    p=hits/users
    denominator=1+z*z/users
    center=(p+z*z/(2*users))/denominator
    radius=z*math.sqrt(p*(1-p)/users+z*z/(4*users*users))/denominator
    return [center-radius,center+radius]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model_service.load_all()
    s=model_service
    bundle=Path(s.ml_dir)
    metrics=json.loads((bundle/'metrics.json').read_text())
    split=json.loads((bundle/'split.json').read_text())
    manifest=json.loads((bundle/'sha256_manifest.json').read_text())
    hashes={}
    for name in ('ncf_baseline.pth','ncf_hybrid.pth','sequential_transformer.pth'):
        actual=hashlib.sha256((bundle/name).read_bytes()).hexdigest()
        assert actual==manifest[name],name
        hashes[name]=actual
    validation=set(split['validation_users']);test=set(split['test_users'])
    assert not validation & test
    for uid,history in s.user_sequences.items():
        assert split['validation_targets'][str(uid)] not in history
        assert split['test_targets'][str(uid)] not in history
    rows=[json.loads(line) for line in (bundle/'test_catalogue.jsonl').read_text().splitlines()]
    aggregate={}
    for name,reported in metrics['full_catalogue_test'].items():
        computed={metric:sum(row['systems'][name][metric] for row in rows)/len(rows)
                  for metric in ('hr10','ndcg10','recall100')}
        assert all(math.isclose(computed[k],reported[k],abs_tol=1e-12) for k in computed)
        hits=sum(row['systems'][name]['hr10'] for row in rows)
        aggregate[name]={'users':len(rows),'hits':hits,**computed,'hr10_95_percent_wilson':wilson(hits,len(rows))}
    arrays=np.load(bundle/'evaluation_candidates.npz')
    users=arrays['test_users'];candidates=arrays['test_candidates']
    targets={int(uid):target for uid,target in ((key,value) for key,value in split['test_targets'].items())}
    models={'baseline':s.ncf_baseline,'hybrid':s.ncf_hybrid,'transformer':s.sequential_transformer}
    sampled={}
    selected={}
    for name,model in models.items():
        actual=evaluate(model,name,users,targets,s.user_sequences,s.movie_genre_matrix,s.user_genre_matrix,candidates,s.device)
        assert all(math.isclose(actual[k],metrics['sampled_test'][name][k],abs_tol=1e-12) for k in ('hr10','ndcg10','mrr')),(name,actual)
        hits=round(actual['hr10']*len(users))
        sampled[name]={**actual,'hits':hits,'hr10_95_percent_wilson':wilson(hits,len(users))}
        best=max(metrics['epochs'][name],key=lambda row:row['validation']['ndcg10'])
        selected[name]={key:best[key] for key in ('epoch','bce','loss')}
    policies=json.loads((bundle/'ranking_policy_metrics.json').read_text())
    policy_rows=[json.loads(line) for line in (bundle/'ranking_policy_per_user.jsonl').read_text().splitlines()]
    for partition in ('validation','test'):
        group=[row for row in policy_rows if row['split']==partition]
        for name,expected in policies[partition].items():
            ranks=[row['top10'][name].index(row['target_idx'])+1 if row['target_idx'] in row['top10'][name] else 0 for row in group]
            hr=sum(rank>0 for rank in ranks)/len(ranks)
            ndcg=sum(1/math.log2(rank+1) for rank in ranks if rank)/len(ranks)
            assert math.isclose(hr,expected['hr10'],abs_tol=1e-12)
            assert math.isclose(ndcg,expected['ndcg10'],abs_tol=1e-12)
    result={'audit_date':'2026-10-14','checkpoint_hashes':hashes,'training_users':len(s.user_sequences),
        'training_interactions':sum(map(len,s.user_sequences.values())),'holdout_exclusion_checked_for_every_training_user':True,
        'validation_test_users_disjoint':True,'original_sampled_metrics_recomputed':sampled,
        'original_catalogue_metrics_recomputed':aggregate,'selected_checkpoints':selected,
        'original_ranking_policy_metrics_verified':True,
        'interval_scope':'Wilson intervals over evaluation users with one frozen model seed; not uncertainty over training runs',
        'source_paths':['metrics.json','split.json','test_catalogue.jsonl','ranking_policy_per_user.jsonl','evaluation_candidates.npz','sha256_manifest.json']}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
