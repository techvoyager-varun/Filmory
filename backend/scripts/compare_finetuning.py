"""Compare parent and continuation on IDENTICAL validation data, not test data."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from train_models import evaluate, padded, negatives, pair_keys, sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--child',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);device=torch.device('cpu')
    root=a.child;config=json.loads((root/'model_config.json').read_text())
    with (root/'user_sequences.pkl').open('rb') as f:histories=pickle.load(f)
    split=json.loads((root/'split.json').read_text());targets={int(k):v for k,v in split['validation_targets'].items()}
    candidates=np.load(root/'evaluation_candidates.npz');users=candidates['val_users'];items=candidates['val_candidates']
    genres=torch.load(root/'movie_genre_matrix.pt',weights_only=True);profiles=torch.load(root/'user_genre_matrix.pt',weights_only=True)
    rng=np.random.default_rng(903);all_users=np.asarray(sorted(histories));probe_users=rng.choice(all_users,8000)
    probe_items=np.asarray([rng.choice(histories[int(u)][1:]) for u in probe_users])
    probe_candidates=np.concatenate([probe_items[:,None],negatives(probe_users,4,len(genres),pair_keys(histories,len(genres)),rng)],axis=1)
    result={}
    for label,bundle in [('parent',a.parent),('finetuned',a.child)]:
        results={}
        for kind in ['baseline','hybrid','transformer']:
            if kind=='baseline':model=NCFBaseline(config['num_users'],len(genres),config['embedding_dim_ncf'])
            elif kind=='hybrid':model=NCFHybrid(config['num_users'],len(genres),len(genres[0]),config['embedding_dim_ncf'])
            else:model=SequentialTransformer(len(genres),config['transformer_embedding_dim'],config['transformer_heads'],config['transformer_layers'],config['transformer_max_len'],pooling='last',dim_feedforward=config['transformer_dim_feedforward'])
            filename='sequential_transformer.pth' if kind=='transformer' else 'ncf_'+kind+'.pth'
            model.load_state_dict(torch.load(bundle/filename,weights_only=True));model.eval()
            results[kind]=evaluate(model,kind,users,targets,histories,genres,profiles,items,device)
            # Fixed validation BCE with one positive and four negatives; no class-ratio change.
            validation_losses=[];probe_losses=[]
            for population,us,cands,destination in [('validation',users,items[:,:5],validation_losses),('training',probe_users,probe_candidates,probe_losses)]:
                for start in range(0,len(us),128):
                    uid=torch.as_tensor(us[start:start+128]);cand=torch.as_tensor(cands[start:start+128]);u=uid[:,None].expand_as(cand)
                    if kind=='transformer':
                        if population=='training':
                            prefix=[histories[int(x)][:histories[int(x)].index(int(y))] for x,y in zip(uid,cand[:,0])]
                        else:prefix=[histories[int(x)] for x in uid]
                        scores=model.score_items(torch.as_tensor(padded(prefix,len(genres),model.max_len)),cand)
                    elif kind=='hybrid':scores=model.logits(u,cand,profiles[u],genres[cand])
                    else:scores=model.logits(u,cand)
                    truth=torch.zeros_like(scores);truth[:,0]=1
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,truth,reduction='none').mean(1)
                    destination.extend(loss.tolist())
            results[kind]['validation_bce_fixed4neg']=float(np.mean(validation_losses))
            results[kind]['training_probe_bce_fixed4neg']=float(np.mean(probe_losses))
            results[kind]['checkpoint_sha256']=sha(bundle/filename)
        result[label]=results
    result['protocol']='Identical 400-user validation ranking and one-positive/four-negative BCE; identical 8000 training probes in eval mode; no test-based selection. Transformer training probe excludes its target from the input prefix.'
    (root/'finetuning_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
