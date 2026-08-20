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

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        x = torch.cat([u_emb, i_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x)).squeeze(-1)

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

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_genre: torch.Tensor,
        item_genre: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        ug_proj = self.user_genre_projection(user_genre)
        ig_proj = self.item_genre_projection(item_genre)
        x = torch.cat([u_emb, i_emb, ug_proj, ig_proj], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x)).squeeze(-1)

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
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        # sequence_tensor: (batch_size, seq_len)
        batch_size, seq_len = sequence_tensor.shape
        positions = torch.arange(seq_len, device=sequence_tensor.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(sequence_tensor) + self.position_embedding(positions)
        mask = sequence_tensor == 0
        out = self.transformer(x, src_key_padding_mask=mask)
        out = self.layer_norm(out)
        return out[:, -1, :]  # return last state embedding (batch_size, embedding_dim)

    def score_candidates_with_sequence(
        self,
        sequence_item_indices: list[int],
        candidate_item_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Calculates similarity between the user sequence representation and candidate items.
        sequence_item_indices: 1-based indices (0 is padding)
        """
        # Trim / pad sequence to max_len
        seq = sequence_item_indices[-self.max_len :] if sequence_item_indices else []
        if len(seq) < self.max_len:
            seq = [0] * (self.max_len - len(seq)) + seq
        
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        seq_emb = self.forward(seq_tensor) # (1, 64)
        
        # Candidate item embeddings (1-indexed in transformer embedding table)
        cand_tensor = candidate_item_indices.to(device) + 1  # 0 is padding, so +1
        cand_embeddings = self.item_embedding(cand_tensor)   # (K, 64)
        
        # Cosine similarity / dot product scoring
        norm_seq = F.normalize(seq_emb, p=2, dim=-1)
        norm_cand = F.normalize(cand_embeddings, p=2, dim=-1)
        scores = torch.mm(norm_cand, norm_seq.t()).squeeze(-1) # (K,)
        # Rescale [-1, 1] to [0, 1]
        return torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0)
