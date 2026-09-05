/**
 * MovieLens-style domain types.
 * These mirror the shapes the future FastAPI backend will return, so switching
 * from mock data to REST endpoints requires no component changes.
 */

export interface Movie {
  movieId: number;
  title: string;
  genres: string[];
  year: number;
  rating: number;
  runtime: number;
  posterUrl: string;
  backdropUrl: string;
  description: string;
}

/** Shape returned by GET /recommendations/:userId and GET /similar/:movieId */
export interface Recommendation {
  movieId: number;
  title: string;
  score: number;
}

/** A recommendation joined with its movie metadata, ready for the UI. */
export interface ScoredMovie extends Movie {
  score?: number;
  /** Stage 1 — NCF Hybrid expert score */
  ncfScore?: number;
  /** Stage 2 — Sequential Transformer expert score */
  transformerScore?: number;
  /** Stage 3 — genre-affinity expert score */
  genreScore?: number;
  /** Stage 4 (DAMR) transparency fields */
  momentumScore?: number;
  agreementScore?: number;
  qualityScore?: number;
  diversityPenalty?: number;
  expertWeights?: ExpertWeights;
  userState?: UserStateInfo;
  variant?: "damr" | "mmr" | "static";
  rank?: number;
  listDiversity?: number;
}

/** Output of the DAMR drift-adaptive expert gate (sums to 1). */
export interface ExpertWeights {
  ncf: number;
  transformer: number;
  genre: number;
}

/** The four interpretable user-state scalars DAMR estimates per request. */
export interface UserStateInfo {
  drift: number;
  focus: number;
  maturity: number;
  freshness: number;
  nInteractions: number;
  weights: ExpertWeights;
}

/** Live taste-state payload from GET /api/taste-state. */
export interface TasteState {
  userId: string;
  modelMapped: boolean;
  state: UserStateInfo;
  vectors: {
    long: number[];
    short: number[];
    momentum: number[];
  };
  genres: string[];
  weights: ExpertWeights;
}

/** Offline evaluation results from GET /api/metrics (ml/metrics.json). */
export interface ModelMetrics {
  protocol: string;
  num_users_evaluated: number;
  num_negatives: number;
  k: number;
  seed: number;
  generated_at: string;
  dataset: {
    num_users_total: number;
    num_items: number;
    num_genres: number;
  };
  /** model name -> { "HR@10": ..., "NDCG@10": ..., MRR, AUC, "ILD@10", "Coverage@10" } */
  results: Record<string, Record<string, number | null>>;
}

export type InteractionType = "play" | "like" | "unlike" | "list_add" | "list_remove";

export interface Interaction {
  userId: string;
  movieId: number;
  type: InteractionType;
  timestamp: string;
}

export interface User {
  userId: string;
  name: string;
  email: string;
  favoriteGenres: string[];
  favoriteMovieIds: number[];
}

export interface AuthCredentials {
  email: string;
  password: string;
}

export interface RegisterPayload {
  name: string;
  email: string;
  password: string;
}
