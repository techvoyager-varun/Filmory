"""Compare rerankers on an existing CLEAN bundle; select on validation only."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.ml.model_service import model_service
from app.ml.damr import damr_rerank, estimate_user_state_from_positions

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args(); bundle = a.bundle.resolve()
    model_service.ml_dir = str(bundle); model_service.load_all()
    torch.set_num_threads(4)
    s = model_service; split = json.loads((bundle/'split.json').read_text())
    all_items = torch.arange(len(s.movie2idx), device=s.device)
    metrics = {}; records = []
    policies = ['retriever', 'normalized_fixed', 'damr']
    for partition in ['validation', 'test']:
        rows = []
        targets = split[partition+'_targets']
        for uid in split[partition+'_users'][:200]:
            hist = s.user_sequences[uid]; target = targets[str(uid)]
            tr = s.sequential_transformer.score_candidates_with_sequence(hist, all_items, s.device)
            ncf = s.ncf_hybrid.score_candidate_items(uid, all_items, s.user_genre_matrix[uid], s.movie_genre_matrix, s.device)
            retriever = tr.clone(); retriever[hist] = -torch.inf
            c = retriever.topk(100).indices
            genre = s.movie_genre_matrix[c] @ s.user_genre_matrix[uid]
            profile = estimate_user_state_from_positions(hist, s.user_genre_matrix[uid])
            rankings = {'retriever': c[:10].tolist()}
            for policy in policies[1:]:
                ranked = damr_rerank(c, ncf[c], tr[c], genre, profile, [None]*100, [None]*100, top_k=10,
                    use_adaptive=policy=='damr', use_momentum=policy=='damr', use_agreement=policy=='damr',
                    use_quality=False, use_diversity=policy=='damr')
                rankings[policy] = [r['modelIdx'] for r in ranked]
            row = {'split': partition, 'user_idx': uid, 'target_idx': target, 'top10': rankings}
            rows.append(row); records.append(row)
        metrics[partition] = {}
        for policy in policies:
            ranks = [r['top10'][policy].index(r['target_idx'])+1 if r['target_idx'] in r['top10'][policy] else 0 for r in rows]
            metrics[partition][policy] = {'users':len(rows), 'hr10':sum(x>0 for x in ranks)/len(rows),
                'ndcg10':sum(1/math.log2(x+1) for x in ranks if x)/len(rows)}
        if partition == 'validation':
            selected = max(policies, key=lambda k:metrics[partition][k]['ndcg10'])
    metrics['selected_on_validation'] = selected
    metrics['limitations'] = ['Static train-prefix context for both holdouts', 'Positional drift proxy; no timestamps',
        'Quality prior disabled; no train-only catalogue rating counts', 'Single seed and 200 users per split']
    (bundle/'ranking_policy_metrics.json').write_text(json.dumps(metrics, indent=2))
    (bundle/'ranking_policy_per_user.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(metrics, indent=2))
if __name__ == '__main__': main()
