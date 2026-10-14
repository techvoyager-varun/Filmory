"""Compare legacy new-user neighbour retrieval with the frozen sequence policy."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
from app.database import Base
from app.models.db_models import Movie
from app.ml.model_service import model_service
from app.ml import recommender


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    args=parser.parse_args()
    output=args.evaluation/'legacy-cold-start-comparison.json'
    if output.exists():
        raise FileExistsError(output)
    report=json.loads((args.evaluation/'metrics.json').read_text())
    torch.set_num_threads(4)
    model_service.load_all()
    recommender.ensure_movie_posters=lambda *args:None
    bundle=Path(model_service.ml_dir)
    split=json.loads((bundle/'split.json').read_text())
    engine=create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        with (bundle/'movies_metadata.csv').open(encoding='utf-8-sig',newline='') as source:
            db.bulk_insert_mappings(Movie,[dict(movie_id=int(row['movieId']),title=row['title'],genres=row['genres'].split('|')) for row in csv.DictReader(source)])
        db.commit()
        records=[]
        for index,uid in enumerate(report['new_test_users']):
            history=model_service.user_sequences[uid]
            genres=model_service.user_genre_matrix[uid]
            favorites=[model_service.idx2genre[int(i)] for i in torch.topk(genres,min(3,len(genres))).indices if float(genres[i])>0]
            seen=[int(model_service.idx2movie[i]) for i in history]
            movies=recommender.get_cold_start_recommendations(favorites,seen,db,top_k=10)
            target=int(model_service.idx2movie[split['test_targets'][str(uid)]])
            top=[m.movieId for m in movies]
            records.append(dict(user_idx=uid,rank=top.index(target)+1 if target in top else 0,top10=top))
            if (index+1)%50==0: print(f"Legacy cold-start: {index+1}/{len(report['new_test_users'])}",flush=True)
    summary={'users':len(records),'hr10':sum(r['rank']>0 for r in records)/len(records),
        'ndcg10':sum(1/math.log2(r['rank']+1) for r in records if r['rank'])/len(records),
        'protocol':'Same additional test users; simulate accounts without learned user embeddings; replay existing train histories and their top-three genres',
        'limitations':['History replay is not a real-new-user experiment','No live satisfaction measurements','Metadata-only disposable catalogue; no synthetic ratings used for ranking'],
        'records':records}
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)
    engine.dispose()


if __name__=='__main__': main()
