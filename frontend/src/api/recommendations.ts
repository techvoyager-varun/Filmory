import { request } from "./client";
import type { ScoredMovie } from "@/types/movie";

/** GET /api/recommendations/:userId */
export async function getRecommendations(userId: string): Promise<ScoredMovie[]> {
  const target = userId && userId !== "guest" ? userId : "me";
  return request<ScoredMovie[]>(`/api/recommendations/${target}?candidate_k=100&top_k=12`);
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
  return request<ScoredMovie[]>(`/api/movies-by-genre?genre=${encodeURIComponent(genre)}&limit=${limit}`);
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
