"""Fresh, reproducible training from ordered zero-based histories or MovieLens CSV.

Never fine-tunes shipped checkpoints. Outputs an isolated bundle and real logs.
Run `python scripts/train_models.py --help` from backend for budget controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(source, ratings_csv=None):
    mappings = {}
    for name in ["user2idx", "idx2user", "movie2idx", "idx2movie", "genre2idx", "idx2genre"]:
        with (source / f"{name}.pkl").open("rb") as f:
            mappings[name] = pickle.load(f)
    with (source / "user_sequences.pkl").open("rb") as f:
        sequences = pickle.load(f)
    lineage = {"kind": "ordered stored training histories", "source_sha256": sha(source / "user_sequences.pkl"),
               "timestamps_available": False,
               "note": "Original notebook test ratings are absent from these histories. New holdouts are taken before fresh training; chronology is ordinal only."}
    if (source / "source_provenance.json").exists():
        lineage = json.loads((source / "source_provenance.json").read_text(encoding="utf-8"))
    if ratings_csv:
        import pandas as pd
        frame = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "timestamp"])
        frame = frame[frame.userId.isin(mappings["user2idx"]) & frame.movieId.isin(mappings["movie2idx"])]
        frame = frame.sort_values(["userId", "timestamp", "movieId"], kind="stable")
        sequences = {mappings["user2idx"][int(u)]: [mappings["movie2idx"][int(i)] for i in g.movieId]
                     for u, g in frame.groupby("userId", sort=True)}
        lineage = {"kind": "MovieLens ratings CSV with existing selected-user/item maps", "source_sha256": sha(ratings_csv),
                   "timestamps_available": True, "note": "Timestamp ordering; tied timestamps broken by movie ID."}
    genres = torch.load(source / "movie_genre_matrix.pt", map_location="cpu", weights_only=True).float()
    return mappings, sequences, genres, lineage


def split_sequences(sequences, num_items):
    train, val, test = {}, {}, {}
    for u, seq in sorted(sequences.items()):
        seq = [int(i) for i in seq]
        if len(seq) < 5:
            continue
        if not all(0 <= i < num_items for i in seq):
            raise ValueError("Sequence contains a canonical ID instead of a model index")
        if len(set(seq)) != len(seq):
            raise ValueError("Repeated ratings need explicit deduplication before leave-two-out")
        train[int(u)], val[int(u)], test[int(u)] = seq[:-2], seq[-2], seq[-1]
    return train, val, test


def pair_keys(histories, num_items):
    return np.sort(np.concatenate([np.asarray(s, dtype=np.int64) + int(u)*num_items for u, s in histories.items()]))


def negatives(users, count, num_items, excluded, rng):
    users = np.asarray(users, dtype=np.int64)
    shape = (len(users), count)
    out = rng.integers(num_items, size=shape, dtype=np.int64)
    for _ in range(100):
        keys = users[:, None]*num_items + out
        pos = np.searchsorted(excluded, keys)
        bad = (pos < len(excluded)) & (excluded[np.minimum(pos, len(excluded)-1)] == keys)
        if not bad.any():
            return out
        out[bad] = rng.integers(num_items, size=int(bad.sum()))
    raise RuntimeError("Negative rejection sampler exhausted its retry budget")


def padded(histories, padding_idx, max_len):
    result = np.full((len(histories), max_len), padding_idx, dtype=np.int64)
    for row, hist in enumerate(histories):
        h = hist[-max_len:]
        if h:
            result[row, -len(h):] = h
    return result


def rank_metrics(scores, top_k=10):
    # Candidates have target in column zero. Conservative ties: negatives tied
    # with target precede it. No arbitrary optimistic tie credit.
    ranks = 1 + (scores[:, 1:] >= scores[:, :1]).sum(1)
    hit = ranks <= top_k
    return {"users": int(len(ranks)), "hr10": float(hit.mean()),
            "ndcg10": float(np.where(hit, 1/np.log2(ranks+1), 0).mean()),
            "mrr": float((1/ranks).mean())}


@torch.inference_mode()
def evaluate(model, kind, users, targets, histories, genres, profiles, candidates, device, batch=128):
    model.eval()
    results = []
    for start in range(0, len(users), batch):
        us = users[start:start+batch]
        cand = torch.as_tensor(candidates[start:start+batch], device=device)
        u = torch.as_tensor(us, device=device)[:, None].expand_as(cand)
        if kind == "transformer":
            seq = torch.as_tensor(padded([histories[int(x)] for x in us], model.padding_idx, model.max_len), device=device)
            scores = model.score_items(seq, cand)
        elif kind == "hybrid":
            scores = model.logits(u, cand, profiles[u], genres[cand])
        else:
            scores = model.logits(u, cand)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite validation predictions")
        results.append(scores.cpu().numpy())
    return rank_metrics(np.concatenate(results))


@torch.inference_mode()
def catalogue_evaluate(models, users, targets, histories, genres, profiles, popularity, device):
    num_items = len(genres)
    items = torch.arange(num_items, device=device)
    per_user = []
    for user in users:
        uid = int(user)
        row = {"user_idx": uid, "target_idx": targets[uid], "systems": {}}
        raw = {"popularity": torch.as_tensor(popularity, device=device).float()}
        for kind, model in models.items():
            model.eval()
            u = torch.full((num_items,), uid, device=device, dtype=torch.long)
            if kind == "hybrid":
                s = model(u, items, profiles[uid].expand(num_items, -1), genres)
            elif kind == "transformer":
                s = model.score_candidates_with_sequence(histories[uid], items, device)
            else:
                s = model(u, items)
            raw[kind] = s
        for kind, scores in raw.items():
            scores = scores.clone()
            scores[histories[uid]] = -torch.inf
            target_score = scores[targets[uid]]
            rank = int((scores >= target_score).sum())
            row["systems"][kind] = {"rank": rank, "hr10": int(rank <= 10), "recall100": int(rank <= 100),
                                    "ndcg10": 1/math.log2(rank+1) if rank <= 10 else 0,
                                    "top10": scores.topk(10).indices.cpu().tolist()}
        per_user.append(row)
    aggregate = {name: {"users": len(users), **{metric: float(np.mean([r["systems"][name][metric] for r in per_user]))
                                              for metric in ["hr10", "recall100", "ndcg10"]}}
                 for name in raw}
    return aggregate, per_user


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BACKEND / "ml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratings-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sequence-epochs", type=int, default=5)
    parser.add_argument("--positives-per-epoch", type=int, default=1000000)
    parser.add_argument("--sequence-samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-users", type=int, default=400)
    parser.add_argument("--catalogue-users", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--test-user-offset", type=int)
    parser.add_argument("--continue-from", type=Path, help="Warm start from a clean bundle with the same pre-training split; never use shipped legacy checkpoints")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    if args.output == args.source or args.source in args.output.parents and args.output.name == "ml":
        parser.error("Training output must be an isolated directory")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "training_source_snapshot.py")
    shutil.copy2(BACKEND / "app/ml/architectures.py", args.output / "architecture_source_snapshot.py")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mappings, original, genre_cpu, lineage = load_data(args.source, args.ratings_csv)
    num_users, num_items = len(mappings["user2idx"]), len(genre_cpu)
    histories, val, test = split_sequences(original, num_items)
    parent = args.continue_from.resolve() if args.continue_from else None
    if parent:
        parent_split = json.loads((parent / "split.json").read_text(encoding="utf-8"))
        parent_config = json.loads((parent / "model_config.json").read_text(encoding="utf-8"))
        if parent_config.get("model_format_version") != 2:
            raise ValueError("Continuation requires a clean version-2 bundle")
        if {str(k): v for k,v in val.items()} != parent_split["validation_targets"] or {str(k): v for k,v in test.items()} != parent_split["test_targets"]:
            raise ValueError("Continuation targets differ from parent; checkpoints could contain holdout information")
        with (parent / "user_sequences.pkl").open("rb") as f:
            if histories != pickle.load(f):
                raise ValueError("Continuation training histories differ from parent")
        if args.embedding_dim != parent_config["embedding_dim_ncf"]:
            raise ValueError("Continuation embedding dimension must match the parent")
        for name in ["user2idx", "movie2idx", "genre2idx"]:
            with (parent / f"{name}.pkl").open("rb") as f:
                if mappings[name] != pickle.load(f): raise ValueError("Continuation mappings differ from parent")
    users_all = np.asarray(sorted(histories), dtype=np.int64)
    if len(users_all) < 2:
        raise ValueError("At least two eligible users are required")
    order = rng.permutation(users_all)
    n_eval = min(args.eval_users, len(order)//2)
    test_offset = args.test_user_offset if args.test_user_offset is not None else n_eval
    if test_offset < n_eval or test_offset+n_eval > len(order):
        parser.error("Test-user offset must leave disjoint, complete evaluation sets")
    val_users, test_users = order[:n_eval], order[test_offset:test_offset+n_eval]
    train_keys = pair_keys(histories, num_items)
    # Training negatives exclude TRAIN ONLY; held-out labels cannot affect fitting.
    # Evaluation negatives exclude all known positives and are sampled without replacement.
    def eval_candidates(us, targets, seed):
        erng = np.random.default_rng(seed)
        rows = []
        for u in us:
            banned = set(histories[int(u)]) | {val[int(u)], test[int(u)]}
            available = np.asarray([i for i in range(num_items) if i not in banned])
            neg = erng.choice(available, size=min(99, len(available)), replace=False)
            rows.append(np.concatenate([[targets[int(u)]], neg]))
        if len({len(x) for x in rows}) != 1:
            raise ValueError("Insufficient unseen evaluation candidates")
        return np.asarray(rows)
    val_candidates = eval_candidates(val_users, val, args.seed+101)
    test_candidates = eval_candidates(test_users, test, args.seed+202)
    np.savez_compressed(args.output / "evaluation_candidates.npz", val_users=val_users, test_users=test_users,
                        val_candidates=val_candidates, test_candidates=test_candidates)
    profile_cpu = torch.zeros(num_users, genre_cpu.shape[1])
    popularity = np.zeros(num_items, dtype=np.int64)
    for uid, hist in histories.items():
        profile_cpu[uid] = genre_cpu[hist].sum(0)
        np.add.at(popularity, hist, 1)
    profile_cpu = F.normalize(profile_cpu, p=1, dim=1)
    genres, profiles = genre_cpu.to(device), profile_cpu.to(device)
    np.save(args.output / "train_popularity.npy", popularity)
    for name, mapping in mappings.items():
        with (args.output / f"{name}.pkl").open("wb") as f:
            pickle.dump(mapping, f)
    for name in ["user_sequences", "user_interacted"]:
        with (args.output / f"{name}.pkl").open("wb") as f:
            pickle.dump(histories, f)
    torch.save(genre_cpu, args.output / "movie_genre_matrix.pt")
    torch.save(profile_cpu, args.output / "user_genre_matrix.pt")
    shutil.copy2(args.source / "movies_metadata.csv", args.output / "movies_metadata.csv")
    (args.output / "split.json").write_text(json.dumps({"seed": args.seed, "validation_targets": val, "test_targets": test,
                                                       "validation_users": val_users.tolist(), "test_users": test_users.tolist()}), encoding="utf-8")
    config = json.loads((args.source / "model_config.json").read_text(encoding="utf-8"))
    config.update(transformer_pooling="last", transformer_dim_feedforward=256, transformer_padding_idx=num_items,
                  model_format_version=2, history_indexing="zero_based", genre_profile_normalization="l1",
                  embedding_dim_ncf=args.embedding_dim)
    (args.output / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    provenance = {"fresh_initialization": True, "source": lineage, "seed": args.seed, "device": str(device),
                  "torch_version": torch.__version__, "eligible_users": len(histories),
                  "training_interactions": sum(map(len, histories.values())),
                  "training_negative_exclusion": "train histories only",
                  "selection": "sampled validation NDCG@10; test not used for selection",
                  "budget": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
                  "architecture_sha256": sha(BACKEND / "app/ml/architectures.py"), "training_script_sha256": sha(__file__)}
    if parent:
        provenance.update(fresh_initialization=False, continuation_from=str(parent),
                          parent_checkpoint_sha256={name:sha(parent/name) for name in ["ncf_baseline.pth", "ncf_hybrid.pth", "sequential_transformer.pth"]},
                          note="Fresh optimizer; continued fitting on identical clean training histories. Original parent checkpoint retained unless validation NDCG improves.")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)
    positive_users = np.repeat(users_all, [len(histories[int(u)]) for u in users_all])
    positive_items = np.concatenate([np.asarray(histories[int(u)]) for u in users_all])
    models = {}
    histories_log = {}
    selected_checkpoints = {}
    for kind in ["baseline", "hybrid", "transformer"]:
        torch.manual_seed(args.seed)
        if kind == "baseline":
            model = NCFBaseline(num_users, num_items, embedding_dim=config.get("embedding_dim_ncf", 8))
        elif kind == "hybrid":
            model = NCFHybrid(num_users, num_items, len(genres[0]), config.get("embedding_dim_ncf", 8))
            # Warm start from this run's CLEAN baseline, not the shipped checkpoint.
            base = models["baseline"]
            model.user_embedding.weight.data.copy_(base.user_embedding.weight.detach().cpu())
            model.item_embedding.weight.data.copy_(base.item_embedding.weight.detach().cpu())
            model.fc1.weight.data.zero_()
            dim = model.user_embedding.embedding_dim
            model.fc1.weight.data[:, :dim].copy_(base.fc1.weight.data[:, :dim].cpu())
            model.fc1.weight.data[:, 2*dim:3*dim].copy_(base.fc1.weight.data[:, dim:].cpu())
            for layer in ["fc1", "fc2", "output"]:
                getattr(model, layer).bias.data.copy_(getattr(base, layer).bias.data.cpu())
            model.fc2.weight.data.copy_(base.fc2.weight.data.cpu())
            model.output.weight.data.copy_(base.output.weight.data.cpu())
        else:
            model = SequentialTransformer(num_items, config.get("transformer_embedding_dim",64),
                                          config.get("transformer_heads",4), config.get("transformer_layers",2),
                                          config.get("transformer_max_len",20), pooling="last", dim_feedforward=256)
            torch.nn.init.normal_(model.item_embedding.weight, std=0.02)
            torch.nn.init.normal_(model.position_embedding.weight, std=0.02)
            model.item_embedding.weight.data[model.padding_idx].zero_()
        model = model.to(device)
        filename = "sequential_transformer.pth" if kind == "transformer" else f"ncf_{kind}.pth"
        if parent:
            model.load_state_dict(torch.load(parent/filename, map_location=device, weights_only=True))
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=1, factor=0.5)
        best, stale = -1.0, 0
        if parent:
            initial = evaluate(model,kind,val_users,val,histories,genres,profiles,val_candidates,device)
            best = initial["ndcg10"]
            torch.save(model.state_dict(), args.output/filename)
            selected_checkpoints[kind] = {"origin":"parent", "validation":initial}
            print(json.dumps({"model":kind,"initial_validation":initial}), flush=True)
        log = []
        filename = "sequential_transformer.pth" if kind == "transformer" else f"ncf_{kind}.pth"
        epochs = args.sequence_epochs if kind == "transformer" else args.epochs
        for epoch in range(1, epochs+1):
            began = time.perf_counter()
            model.train()
            if kind == "transformer":
                # Uniform users rather than the notebook's first-300k interaction
                # cutoff, which omitted most users from sequence training.
                sample_users = rng.choice(users_all, size=args.sequence_samples, replace=True)
                positions = np.asarray([rng.integers(1, len(histories[int(u)])) for u in sample_users])
                xs = padded([histories[int(u)][max(0,p-model.max_len):p] for u,p in zip(sample_users,positions)], num_items, model.max_len)
                ys = np.asarray([histories[int(u)][p] for u,p in zip(sample_users,positions)])
                batch_size = 256
            else:
                chosen = rng.choice(len(positive_items), size=min(args.positives_per_epoch, len(positive_items)), replace=False)
                sample_users, ys = positive_users[chosen], positive_items[chosen]
                batch_size = args.batch_size
            loss_sum, bce_sum, total = 0.0, 0.0, 0
            for start in range(0, len(ys), batch_size):
                us = sample_users[start:start+batch_size]
                cand_np = np.concatenate([ys[start:start+batch_size,None], negatives(us,4,num_items,train_keys,rng)],axis=1)
                cand = torch.as_tensor(cand_np, device=device)
                u = torch.as_tensor(us, device=device)[:,None].expand_as(cand)
                opt.zero_grad(set_to_none=True)
                if kind == "transformer":
                    scores = model.score_items(torch.as_tensor(xs[start:start+batch_size],device=device),cand)
                elif kind == "hybrid":
                    scores = model.logits(u,cand,profiles[u],genres[cand])
                else:
                    scores = model.logits(u,cand)
                labels = torch.zeros_like(scores); labels[:,0] = 1
                bce = F.binary_cross_entropy_with_logits(scores,labels)
                # Joint classification/ranking objective: do not optimize a
                # misleading "accuracy percentage" by predicting negatives only.
                loss = bce + 0.25*F.softplus(scores[:,1:]-scores[:,:1]).mean()
                if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                count=len(us); total+=count; loss_sum+=float(loss.detach())*count; bce_sum+=float(bce.detach())*count
            validation = evaluate(model,kind,val_users,val,histories,genres,profiles,val_candidates,device)
            scheduler.step(validation["ndcg10"])
            record={"model":kind,"epoch":epoch,"loss":loss_sum/total,"bce":bce_sum/total,
                    "positives":total,"validation":validation,"seconds":time.perf_counter()-began,
                    "lr":opt.param_groups[0]["lr"]}
            log.append(record)
            with (args.output/"training_log.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True)
            if validation["ndcg10"] > best:
                best=validation["ndcg10"]; stale=0
                torch.save(model.state_dict(),args.output/filename)
                selected_checkpoints[kind] = {"origin":"this_run", "epoch":epoch, "bce":record["bce"], "loss":record["loss"], "validation":validation}
            else: stale+=1
            if stale>=3: break
        model.load_state_dict(torch.load(args.output/filename,map_location=device,weights_only=True))
        model.eval(); models[kind]=model; histories_log[kind]=log
    sampled={kind:evaluate(m,kind,test_users,test,histories,genres,profiles,test_candidates,device) for kind,m in models.items()}
    # Validation catalogue checks happen before evaluating the test catalogue.
    cv=val_users[:min(args.catalogue_users,len(val_users))]
    ct=test_users[:min(args.catalogue_users,len(test_users))]
    val_full,val_rows=catalogue_evaluate(models,cv,val,histories,genres,profiles,popularity,device)
    preferred=max(models,key=lambda name:(val_full[name]["ndcg10"],val_full[name]["recall100"]))
    config["preferred_retriever"]=preferred
    (args.output/"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    test_full,test_rows=catalogue_evaluate(models,ct,test,histories,genres,profiles,popularity,device)
    for name,records in [("validation_catalogue.jsonl",val_rows),("test_catalogue.jsonl",test_rows)]:
        (args.output/name).write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
    result={"protocol":"fresh leave-two-out training; sampled and full-catalogue metrics reported separately",
            "sampled_test":sampled,"full_catalogue_validation":val_full,"full_catalogue_test":test_full,
            "preferred_retriever_from_validation":preferred,"epochs":histories_log,
            "selected_checkpoints":selected_checkpoints,
            "limitations":["Single seed; bounded per-epoch sampling budget", "No original notebook test ratings unless CSV supplied",
                            "No independent human assessment or publication certification"]}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="epochs"}),flush=True)
    return result


if __name__ == "__main__":
    main()
