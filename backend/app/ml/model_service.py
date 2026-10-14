import os
import json
import pickle
import logging
import hashlib
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Set

from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer

logger = logging.getLogger(__name__)

class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ml_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml")
        pointer = os.path.join(self.ml_dir, "active_bundle.json")
        if os.path.exists(pointer):
            with open(pointer, encoding="utf-8") as f:
                bundle = json.load(f)["path"]
            resolved = os.path.abspath(os.path.join(self.ml_dir, bundle))
            if not os.path.isfile(os.path.join(resolved, "model_config.json")):
                raise RuntimeError("Active model bundle is missing; restore active_bundle.json or retrain")
            self.ml_dir = resolved
        
        # Mappings
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.genre2idx: Dict[str, int] = {}
        self.idx2genre: Dict[int, str] = {}
        
        # Config & matrices
        self.config: Dict[str, Any] = {}
        self.serving_policy: Dict[str, Any] = {}
        self.movie_genre_matrix: Optional[torch.Tensor] = None
        self.user_genre_matrix: Optional[torch.Tensor] = None
        
        # Models
        self.ncf_baseline: Optional[NCFBaseline] = None
        self.ncf_hybrid: Optional[NCFHybrid] = None
        self.sequential_transformer: Optional[SequentialTransformer] = None
        
        # Historical interactions / sequences from training (fallback)
        self.user_interacted: Dict[int, List[int]] = {}
        self.user_sequences: Dict[int, List[int]] = {}
        
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_all(self):
        """Load available recommendation artifacts once and prepare models for inference."""
        if self.is_loaded:
            return
        self.serving_policy = {}
        
        logger.info(f"Loading ML models and artifacts from {self.ml_dir} on device: {self.device}")
        
        # 1. Load config
        config_path = os.path.join(self.ml_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        
        # 2. Load mappings
        def load_pkl(filename: str):
            path = os.path.join(self.ml_dir, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            return {}

        self.user2idx = load_pkl("user2idx.pkl")
        self.idx2user = load_pkl("idx2user.pkl")
        self.movie2idx = load_pkl("movie2idx.pkl")
        self.idx2movie = load_pkl("idx2movie.pkl")
        self.genre2idx = load_pkl("genre2idx.pkl")
        self.idx2genre = load_pkl("idx2genre.pkl")
        self.user_interacted = load_pkl("user_interacted.pkl")
        self.user_sequences = load_pkl("user_sequences.pkl")

        # 3. Load matrices
        mg_path = os.path.join(self.ml_dir, "movie_genre_matrix.pt")
        if os.path.exists(mg_path):
            self.movie_genre_matrix = torch.load(mg_path, map_location=self.device, weights_only=True).float()
        
        ug_path = os.path.join(self.ml_dir, "user_genre_matrix.pt")
        if os.path.exists(ug_path):
            self.user_genre_matrix = torch.load(ug_path, map_location=self.device, weights_only=True).float()

        # 4. Instantiate & load models
        num_users = self.config.get("num_users", 41547)
        num_items = self.config.get("num_items", 22836)
        num_genres = self.config.get("num_genres", 20)
        emb_ncf = self.config.get("embedding_dim_ncf", 8)
        emb_trans = self.config.get("transformer_embedding_dim", 64)
        heads_trans = self.config.get("transformer_heads", 4)
        layers_trans = self.config.get("transformer_layers", 2)
        maxlen_trans = self.config.get("transformer_max_len", 20)

        # Baseline NCF
        baseline_path = os.path.join(self.ml_dir, "ncf_baseline.pth")
        if os.path.exists(baseline_path):
            self.ncf_baseline = NCFBaseline(num_users=num_users, num_items=num_items, embedding_dim=emb_ncf)
            self.ncf_baseline.load_state_dict(torch.load(baseline_path, map_location=self.device, weights_only=True))
            self.ncf_baseline.to(self.device)
            self.ncf_baseline.eval()
            logger.info("Loaded NCF Baseline model")

        # Hybrid NCF
        hybrid_path = os.path.join(self.ml_dir, "ncf_hybrid.pth")
        if os.path.exists(hybrid_path):
            self.ncf_hybrid = NCFHybrid(
                num_users=num_users, num_items=num_items, num_genres=num_genres, embedding_dim=emb_ncf
            )
            self.ncf_hybrid.load_state_dict(torch.load(hybrid_path, map_location=self.device, weights_only=True))
            self.ncf_hybrid.to(self.device)
            self.ncf_hybrid.eval()
            logger.info("Loaded NCF Hybrid model")

        # Sequential Transformer
        trans_path = os.path.join(self.ml_dir, "sequential_transformer.pth")
        if os.path.exists(trans_path):
            self.sequential_transformer = SequentialTransformer(
                num_items=num_items,
                embedding_dim=emb_trans,
                num_heads=heads_trans,
                num_layers=layers_trans,
                max_len=maxlen_trans,
                pooling=self.config.get("transformer_pooling", "legacy_length"),
                dim_feedforward=self.config.get("transformer_dim_feedforward", 2048),
            )
            self.sequential_transformer.load_state_dict(torch.load(trans_path, map_location=self.device, weights_only=True))
            self.sequential_transformer.to(self.device)
            self.sequential_transformer.eval()
            logger.info("Loaded Sequential Transformer model")

        if self.config.get("model_format_version") == 2 and any(x is None for x in [
            self.ncf_baseline, self.ncf_hybrid, self.sequential_transformer,
            self.movie_genre_matrix, self.user_genre_matrix,
        ]):
            raise RuntimeError("Active training bundle is incomplete; restore artifacts before serving")
        # Serving experiments live outside the immutable checkpoint bundle.
        # Reject stale overlays rather than silently applying old tuning to new weights.
        policy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "serving_policy.json")
        if os.path.exists(policy_path):
            with open(policy_path, encoding="utf-8") as f:
                policy = json.load(f)
            for filename in ("ncf_baseline.pth", "ncf_hybrid.pth", "sequential_transformer.pth"):
                with open(os.path.join(self.ml_dir, filename), "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if policy.get("checkpoint_hashes", {}).get(filename) != digest:
                    raise RuntimeError("Serving policy does not match the active checkpoints")
            bins = policy.get("history_bins", [])
            if not bins or len({entry.get("min_history") for entry in bins}) != len(bins):
                raise RuntimeError("Invalid serving policy history bins")
            for entry in bins:
                if (not isinstance(entry.get("min_history"), int) or not 2 <= entry["min_history"] <= maxlen_trans
                    or entry.get("mode") not in {"sequence_only", "legacy_cold_start"}):
                    raise RuntimeError("Invalid serving policy history bin")
                if entry["mode"] == "legacy_cold_start":
                    continue
                weights = entry.get("weights", [])
                if (entry.get("strategy") not in {"transformer", "mixed", "collaborative"}
                    or not isinstance(entry.get("candidate_k"), int)
                    or not 10 <= entry["candidate_k"] <= 1000
                    or len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights)
                    or abs(sum(weights) - 1.0) > 1e-6):
                    raise RuntimeError("Invalid sequence-only serving policy")
            self.serving_policy = policy
        self.is_loaded = True
        logger.info("All ML artifacts loaded and ready for inference.")

model_service = ModelService.get_instance()
