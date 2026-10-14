import torch
import torch.nn as nn
import torch.nn.functional as F

class NCFBaseline(nn.Module):
    def __init__(self, num_users: int = 41547, num_items: int = 22836, embedding_dim: int = 8):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def logits(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        x = torch.cat([u_emb, i_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x).squeeze(-1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(user_idx, item_idx))

    def score_all_items_for_user(self, user_idx_int: int, device: torch.device) -> torch.Tensor:
        """Efficient batch scoring of all items for a single user"""
        num_items = self.item_embedding.num_embeddings
        user_tensor = torch.full((num_items,), user_idx_int, dtype=torch.long, device=device)
        item_tensor = torch.arange(num_items, dtype=torch.long, device=device)
        return self.forward(user_tensor, item_tensor)


class NCFHybrid(nn.Module):
    def __init__(
        self,
        num_users: int = 41547,
        num_items: int = 22836,
        num_genres: int = 20,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_genre_projection = nn.Linear(num_genres, embedding_dim)
        self.item_genre_projection = nn.Linear(num_genres, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def logits(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_genre: torch.Tensor,
        item_genre: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        # Notebook training used a genre probability distribution (L1 norm).
        # Serving also uses L2 profiles for cosine similarity; restore the
        # training scale before feeding the learned genre projection.
        user_genre = F.normalize(user_genre, p=1, dim=-1)
        ug_proj = self.user_genre_projection(user_genre)
        ig_proj = self.item_genre_projection(item_genre)
        # This order is part of the checkpoint contract (notebook cell 71).
        x = torch.cat([u_emb, ug_proj, i_emb, ig_proj], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x).squeeze(-1)

    def forward(self, user_idx, item_idx, user_genre, item_genre) -> torch.Tensor:
        return torch.sigmoid(self.logits(user_idx, item_idx, user_genre, item_genre))

    def score_candidate_items(
        self,
        user_idx_int: int,
        candidate_item_indices: torch.Tensor,
        user_genre_vector: torch.Tensor,
        movie_genre_matrix: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch score specific candidate items for a user"""
        k = len(candidate_item_indices)
        user_tensor = torch.full((k,), user_idx_int, dtype=torch.long, device=device)
        user_genre_expanded = user_genre_vector.unsqueeze(0).expand(k, -1).to(device)
        item_genre_batch = movie_genre_matrix[candidate_item_indices].to(device)
        
        return self.forward(
            user_tensor,
            candidate_item_indices.to(device),
            user_genre_expanded,
            item_genre_batch,
        )


class SequentialTransformer(nn.Module):
    def __init__(
        self,
        num_items: int = 22836,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 20,
        pooling: str = "legacy_length",
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.padding_idx = num_items
        if pooling not in {"legacy_length", "last"}:
            raise ValueError("Unknown sequence pooling convention")
        self.pooling = pooling
        # Movie indices are zero-based. The final row, not row zero, is padding.
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=self.padding_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        # sequence_tensor: (batch_size, seq_len)
        batch_size, seq_len = sequence_tensor.shape
        positions = torch.arange(seq_len, device=sequence_tensor.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(sequence_tensor) + self.position_embedding(positions)
        mask = sequence_tensor == self.padding_idx
        empty = mask.all(dim=1)
        # All-padding attention is undefined. Unmask one position, then zero
        # its representation; callers receive a neutral score for no history.
        safe_mask = mask.clone()
        safe_mask[empty, -1] = False
        out = self.transformer(x, src_key_padding_mask=safe_mask)
        out = self.layer_norm(out)
        if self.pooling == "legacy_length":
            # Reproduce the original checkpoint's training behavior exactly.
            # Its left-padding/length-minus-one mismatch is fixed for NEW runs
            # by selecting pooling="last", rather than silently changing weights.
            positions = (~mask).sum(1).clamp(min=1) - 1
        else:
            positions = torch.arange(seq_len, device=sequence_tensor.device).expand(batch_size, -1)
            positions = positions.masked_fill(mask, -1).max(1).values.clamp(min=0)
        result = out[torch.arange(batch_size, device=sequence_tensor.device), positions]
        return result.masked_fill(empty.unsqueeze(1), 0.0)

    def score_items(self, sequences: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        representation = self.forward(sequences)
        return (representation.unsqueeze(1) * self.item_embedding(candidate_items)).sum(-1)

    def score_candidates_with_sequence(
        self,
        sequence_item_indices: list[int],
        candidate_item_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Calculates similarity between the user sequence representation and candidate items.
        Both history and candidates use zero-based model item indices.
        """
        # Trim / pad sequence to max_len
        seq = sequence_item_indices[-self.max_len :] if sequence_item_indices else []
        if len(seq) < self.max_len:
            seq = [self.padding_idx] * (self.max_len - len(seq)) + seq
        
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        logits = self.score_items(seq_tensor, candidate_item_indices.to(device)).squeeze(0)
        return torch.sigmoid(logits)
