"""Regression checks against inspected training definitions, not just API shapes."""
import ast
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer
from scripts.train_models import split_sequences, negatives, pair_keys, padded


def notebook_class(cell, name):
    path = Path(__file__).resolve().parents[1] / "training/source_notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][cell]["source"])
    tree = ast.parse(source)
    # Execute only the reviewed model class, never notebook downloads, training,
    # installs, outputs, prose, or unrelated cells.
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef)
    assert tree.body[0].name == name
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(compile(tree, str(path), "exec"), ns)
    return ns[name]


def test_baseline_matches_notebook_predictions():
    torch.manual_seed(42)
    reference = notebook_class(29, "NCF")(7, 13).eval()
    served = NCFBaseline(7, 13).eval()
    served.load_state_dict(reference.state_dict())
    u, i = torch.tensor([0, 2, 6]), torch.tensor([0, 7, 12])
    torch.testing.assert_close(served(u, i), reference(u, i))


def test_hybrid_matches_notebook_feature_order_and_scale():
    torch.manual_seed(42)
    reference = notebook_class(71, "HybridNCF")(7, 13, 4).eval()
    served = NCFHybrid(7, 13, 4).eval()
    served.load_state_dict(reference.state_dict())
    u, i = torch.tensor([0, 2, 6]), torch.tensor([0, 7, 12])
    ug = F.normalize(torch.rand(3, 4), p=1, dim=1)
    ig = torch.randint(0, 2, (3, 4)).float()
    torch.testing.assert_close(served(u, i, ug, ig), torch.sigmoid(reference(u, i, ug, ig)))
    torch.testing.assert_close(served(u, i, F.normalize(ug, dim=1), ig), served(u, i, ug, ig))


def test_transformer_matches_notebook_zero_based_logits():
    torch.manual_seed(42)
    reference = notebook_class(101, "SequentialRecommender")(13, max_len=5).eval()
    served = SequentialTransformer(13, max_len=5, pooling="legacy_length").eval()
    served.load_state_dict(reference.state_dict())
    # Include item zero, the highest valid item, left padding, and full history.
    seq = torch.tensor([[13, 13, 0, 7, 12], [0, 1, 2, 3, 4]])
    candidates = torch.tensor([[0, 7, 12], [1, 3, 12]])
    with torch.no_grad():
        torch.testing.assert_close(served.score_items(seq, candidates), reference.score_items(seq, candidates))
        scores = served.score_candidates_with_sequence([0, 7, 12], candidates[0], torch.device("cpu"))
        torch.testing.assert_close(scores, torch.sigmoid(reference.score_items(seq[:1], candidates[:1])).squeeze(0))


def test_new_transformer_pools_last_valid_and_empty_history_is_neutral():
    torch.manual_seed(42)
    model = SequentialTransformer(13, max_len=5, pooling="last").eval()
    seq = torch.tensor([[13, 13, 0, 7, 12]])
    captured = {}
    hook = model.layer_norm.register_forward_hook(lambda _, args, output: captured.update(out=output))
    with torch.no_grad():
        representation = model(seq)
        torch.testing.assert_close(representation, captured["out"][:, -1])
        empty_scores = model.score_candidates_with_sequence([], torch.tensor([0, 12]), torch.device("cpu"))
        torch.testing.assert_close(empty_scores, torch.full((2,), 0.5))
    hook.remove()


def test_fresh_split_and_training_negatives_exclude_training_positives():
    histories, val, test = split_sequences({0: [0, 1, 2, 3, 4], 1: [3, 2, 1, 0, 4]}, 6)
    assert histories[0] == [0, 1, 2] and val[0] == 3 and test[0] == 4
    assert val[0] not in histories[0] and test[0] not in histories[0]
    import numpy as np
    us = np.asarray([0, 1]*20)
    out = negatives(us, 4, 6, pair_keys(histories, 6), np.random.default_rng(42))
    assert all(not (set(items) & set(histories[int(u)])) for u, items in zip(us, out))
    assert padded([[0, 4]], 6, 4).tolist() == [[6, 6, 0, 4]]
