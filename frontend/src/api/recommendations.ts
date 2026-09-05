import { request } from "./client";
import type { ScoredMovie, TasteState, ModelMetrics } from "@/types/movie";

/** GET /api/recommendations/:userId */
export async function getRecommendations(
  userId: string,
  variant: "damr" | "mmr" | "static" = "damr",
): Promise<ScoredMovie[]> {
  const target = userId && userId !== "guest" ? userId : "me";
  return request<ScoredMovie[]>(
    `/api/recommendations/${target}?candidate_k=100&top_k=12&variant=${variant}`,
  );
}

/** GET /api/metrics — offline evaluation (HR@K / NDCG@K / MRR / AUC) */
export async function getModelMetrics(): Promise<ModelMetrics> {
  return request<ModelMetrics>("/api/metrics");
}

/** GET /api/taste-state — live DAMR drift/focus/maturity/freshness + profiles */
export async function getTasteState(): Promise<TasteState> {
  return request<TasteState>("/api/taste-state");
}

/** GET /api/similar/:movieId */
export async function getSimilarMovies(movieId: number): Promise<ScoredMovie[]> {
  return request<ScoredMovie[]>(`/api/similar/${movieId}?top_k=14`);
}

/** GET /api/trending */
export async function getTrending(): Promise<ScoredMovie[]> {
  return request<ScoredMovie[]>("/api/trending?limit=20");
}

/** GET /api/popular */
export async function getPopular(): Promise<ScoredMovie[]> {
  return request<ScoredMovie[]>("/api/popular?limit=20");
}

/** GET /api/movies-by-genre?genre= */
export async function getByGenre(genre: string, limit = 20): Promise<ScoredMovie[]> {
  return request<ScoredMovie[]>(
    `/api/movies-by-genre?genre=${encodeURIComponent(genre)}&limit=${limit}`,
  );
}

/** POST /api/cold-start */
export async function getColdStartRecommendations(
  favoriteGenres: string[],
  favoriteMovieIds: number[],
  limit = 10,
): Promise<ScoredMovie[]> {
  return request<ScoredMovie[]>(`/api/cold-start?top_k=${limit}`, {
    method: "POST",
    body: JSON.stringify({ favoriteGenres, favoriteMovieIds }),
  });
}
