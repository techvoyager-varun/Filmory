import { request } from "./client";
import type { Movie } from "@/types/movie";

export interface MoviePage {
  movies: Movie[];
  total: number;
}

export interface BrowseParams {
  genre?: string;
  sort?: "popular" | "rating" | "year" | "title";
  query?: string;
  offset?: number;
  limit?: number;
}

/** GET /api/genres */
export async function getGenres(): Promise<string[]> {
  return request<string[]>("/api/genres");
}

/** GET /api/movies?genre=&sort=&query=&offset=&limit= */
export async function browseMovies({
  genre = "All",
  sort = "popular",
  query = "",
  offset = 0,
  limit = 48,
}: BrowseParams = {}): Promise<MoviePage> {
  const params = new URLSearchParams({
    genre,
    sort,
    query,
    offset: String(offset),
    limit: String(limit),
  });
  return request<MoviePage>(`/api/movies?${params.toString()}`);
}

/** GET /api/movies */
export async function getMovies(): Promise<Movie[]> {
  const page = await request<MoviePage>("/api/movies?limit=60");
  return page.movies;
}

/** GET /api/movies/:id */
export async function getMovie(movieId: number): Promise<Movie | null> {
  try {
    return await request<Movie>(`/api/movies/${movieId}`);
  } catch {
    return null;
  }
}

/** GET /api/search?query= */
export async function searchMovies(query: string): Promise<Movie[]> {
  const q = query.trim();
  if (!q) return [];
  return request<Movie[]>(`/api/search?query=${encodeURIComponent(q)}`);
}

export async function getMoviesByIds(ids: number[]): Promise<Movie[]> {
  if (!ids.length) return [];
  const promises = ids.map((id) => getMovie(id));
  const results = await Promise.all(promises);
  return results.filter((m): m is Movie => m !== null);
}
