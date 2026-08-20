import { request } from "./client";
import type { Interaction, InteractionType, Movie } from "@/types/movie";

export interface HistoryEntry {
  movie: Movie;
  watchedAt: string;
  progress: number;
}

/** POST /api/interactions */
export async function recordInteraction(data: {
  userId?: string;
  movieId: number;
  type: InteractionType;
}): Promise<Interaction> {
  return request<Interaction>("/api/interactions", {
    method: "POST",
    body: JSON.stringify({ movieId: data.movieId, type: data.type }),
  });
}

/** GET /api/users/me/history */
export async function getHistory(_userId?: string): Promise<HistoryEntry[]> {
  try {
    return await request<HistoryEntry[]>("/api/users/me/history");
  } catch {
    return [];
  }
}

/** GET /api/users/me/my-list */
export async function getMyList(_userId?: string): Promise<Movie[]> {
  try {
    return await request<Movie[]>("/api/users/me/my-list");
  } catch {
    return [];
  }
}

/** POST /api/users/me/my-list/:movieId */
export async function addToMyList(_userId: string, movieId: number): Promise<Movie[]> {
  return request<Movie[]>(`/api/users/me/my-list/${movieId}`, {
    method: "POST",
  });
}

/** DELETE /api/users/me/my-list/:movieId */
export async function removeFromMyList(_userId: string, movieId: number): Promise<Movie[]> {
  return request<Movie[]>(`/api/users/me/my-list/${movieId}`, {
    method: "DELETE",
  });
}

/** GET /api/users/me/likes */
export async function getLikes(): Promise<number[]> {
  try {
    return await request<number[]>("/api/users/me/likes");
  } catch {
    return [];
  }
}

/** POST /api/users/me/likes/:movieId */
export async function toggleLike(_userId: string, movieId: number): Promise<number[]> {
  return request<number[]>(`/api/users/me/likes/${movieId}`, {
    method: "POST",
  });
}
