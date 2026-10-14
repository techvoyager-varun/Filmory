"""Frozen-model candidate retrieval and ranking shared by serving and evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def choose_history_policy(policy: dict, history_length: int) -> dict:
    eligible = [entry for entry in policy.get("history_bins", []) if entry["min_history"] <= history_length]
    return max(eligible, key=lambda entry:entry["min_history"]) if eligible else {"mode":"legacy_cold_start"}


@torch.no_grad()
def score_neighbor_items(context: list[int], profile: torch.Tensor, service) -> torch.Tensor:
    """Item-only collaborative votes; no learned embedding of the requesting user."""
    item_count = len(service.movie2idx)
    if not context or service.user_genre_matrix is None or service.movie_genre_matrix is None:
        return torch.zeros(item_count, device=service.device)
    target = service.movie_genre_matrix[context].sum(0).clone()
    for index in profile.topk(min(3, len(profile))).indices:
        if float(profile[index]) > 0:
            target[index] += 2.0
    target = F.normalize(target, dim=0)
    similarities = F.cosine_similarity(service.user_genre_matrix, target.unsqueeze(0), dim=-1)
    neighbors = similarities.topk(min(100, len(similarities))).indices.tolist()
    votes = torch.zeros(item_count, device=service.device)
    for uid in neighbors:
        items = torch.tensor(service.user_interacted.get(uid, []), dtype=torch.long, device=service.device)
        votes.scatter_add_(0, items, torch.ones(len(items), device=service.device) * similarities[uid])
    alignment = service.movie_genre_matrix @ target
    scores = votes * (1.0 + 4.0 * alignment)
    scores[alignment <= 0] = 0.0
    return scores


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if not scores.numel():
        return scores.clone()
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if float(span) > 1e-8 else torch.zeros_like(scores)


def select_candidate_positions(
    transformer: torch.Tensor,
    ncf: torch.Tensor,
    genre: torch.Tensor,
    candidate_k: int,
    strategy: str = "transformer",
) -> torch.Tensor:
    """Return unique pool positions; inputs contain eligible items only."""
    count = min(max(0, candidate_k), transformer.numel())
    if not count:
        return torch.empty(0, dtype=torch.long, device=transformer.device)
    if strategy == "transformer":
        return torch.topk(transformer, count).indices
    if strategy == "collaborative":
        return torch.topk(ncf, count).indices
    if strategy != "mixed":
        raise ValueError(f"Unknown candidate strategy: {strategy}")
    # Reserve half the pool for sequence retrieval, then interleave other
    # experts. Fill from the sequence ordering if the experts overlap heavily.
    orderings = [torch.topk(signal, count).indices.tolist() for signal in (transformer, ncf, genre)]
    chosen = orderings[0][:max(1, count // 2)]
    seen = set(chosen)
    for row in zip(orderings[1], orderings[2], orderings[0]):
        for position in row:
            if position not in seen:
                chosen.append(position)
                seen.add(position)
            if len(chosen) == count:
                return torch.tensor(chosen, dtype=torch.long, device=transformer.device)
    return torch.tensor(chosen, dtype=torch.long, device=transformer.device)


def rank_candidate_positions(
    positions: torch.Tensor,
    transformer: torch.Tensor,
    ncf: torch.Tensor,
    genre: torch.Tensor,
    weights: list[float] | tuple[float, float, float],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights are NCF, Transformer, genre; normalization uses the chosen pool."""
    if len(weights) != 3 or any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Ranking weights must be nonnegative and sum to one")
    scores = sum(weight * normalize_scores(signal[positions])
                 for weight, signal in zip(weights, (ncf, transformer, genre)))
    values, order = torch.topk(scores, min(max(0, top_k), len(positions)))
    return positions[order], values
