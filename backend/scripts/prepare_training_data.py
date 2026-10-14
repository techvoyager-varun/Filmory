"""Build a zero-based source bundle from original MovieLens CSV files.

The source bundle is NOT a trained serving bundle. train_models.py performs
leave-two-out BEFORE fitting models or user profiles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def prepare(ratings_csv, movies_csv, output, fraction=0.30, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ratings = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "timestamp"])
    movies = pd.read_csv(movies_csv, usecols=["movieId", "title", "genres"]).fillna({"genres":"(no genres listed)"})
    users = ratings.userId.unique()
    if not 0 < fraction <= 1: raise ValueError("User fraction must be in (0,1]")
    selected = np.random.RandomState(seed).choice(users, size=max(1,int(len(users)*fraction)), replace=False)
    ratings = ratings[ratings.userId.isin(selected)].sort_values(["userId","timestamp","movieId"],kind="stable")
    if ratings.duplicated(["userId","movieId"]).any():
        raise ValueError("Duplicate user/movie ratings need explicit preprocessing")
    user2idx={int(u):i for i,u in enumerate(sorted(ratings.userId.unique()))}
    movie2idx={int(m):i for i,m in enumerate(sorted(ratings.movieId.unique()))}
    genres=sorted({g for s in movies.genres for g in s.split("|")})
    genre2idx={g:i for i,g in enumerate(genres)}
    mappings={"user2idx":user2idx,"movie2idx":movie2idx,"genre2idx":genre2idx,
              "idx2user":{i:u for u,i in user2idx.items()},"idx2movie":{i:m for m,i in movie2idx.items()},
              "idx2genre":{i:g for g,i in genre2idx.items()}}
    for name,mapping in mappings.items():
        with (output/f"{name}.pkl").open("wb") as f:pickle.dump(mapping,f)
    sequences={user2idx[int(u)]:[movie2idx[int(m)] for m in group.movieId] for u,group in ratings.groupby("userId",sort=True)}
    with (output/"user_sequences.pkl").open("wb") as f:pickle.dump(sequences,f)
    matrix=torch.zeros(len(movie2idx),len(genres))
    for row in movies.itertuples(index=False):
        if int(row.movieId) in movie2idx:
            for g in row.genres.split("|"):matrix[movie2idx[int(row.movieId)],genre2idx[g]]=1
    if (matrix.sum(1)==0).any():raise ValueError("Mapped items are missing movie metadata")
    torch.save(matrix,output/"movie_genre_matrix.pt")
    movies.to_csv(output/"movies_metadata.csv",index=False)
    config={"num_users":len(user2idx),"num_items":len(movie2idx),"num_genres":len(genres),
            "embedding_dim_ncf":16,"transformer_embedding_dim":64,"transformer_heads":4,
            "transformer_layers":2,"transformer_max_len":20,"candidate_k":100,"top_k":10,
            "ncf_weight":0.55,"transformer_weight":0.25,"genre_weight":0.20}
    (output/"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    provenance={"kind":"original MovieLens CSV source bundle before splitting", "timestamps_available":True,
                "seed":seed,"user_fraction":fraction,"selected_interactions":len(ratings),
                "source_sha256":{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [ratings_csv,movies_csv]},
                "note":"Sequences retain timestamp order (ties broken by movieId); split before training. Metadata maps include the selected transductive catalogue."}
    (output/"source_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    return provenance


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings-csv",type=Path,required=True);p.add_argument("--movies-csv",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--user-fraction",type=float,default=.30)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();print(json.dumps(prepare(a.ratings_csv,a.movies_csv,a.output,a.user_fraction,a.seed),indent=2))
