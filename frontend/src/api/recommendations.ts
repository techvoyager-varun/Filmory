import { loadCatalog, type CatalogMovie } from "@/data/catalog";
import type { ScoredMovie } from "@/types/movie";

/**
 * Placeholder ranking derived from catalog signals (rating, popularity, genre
 * overlap). The FastAPI + PyTorch service will return the identical
 * `{ movieId, title, score }` shape, so the UI needs no changes.
 */

function scoreOf(movie: CatalogMovie, base: number): number {
  return Number(Math.min(0.99, base + movie.rating / 40).toFixed(2));
}

/** GET /recommendations/:userId */
export async function getRecommendations(_userId: string): Promise<ScoredMovie[]> {
  const { featured, topRated } = await loadCatalog();
  const pool = featured.length >= 10 ? featured : topRated;
  return pool.slice(0, 12).map((movie, i) => ({ ...movie, score: scoreOf(movie, 0.72 - i * 0.01) }));
}

/** GET /similar/:movieId */
export async function getSimilarMovies(movieId: number): Promise<ScoredMovie[]> {
  const { byId, topRated } = await loadCatalog();
  const movie = byId.get(movieId);
  if (!movie) return [];

  return topRated
    .filter((m) => m.movieId !== movieId)
    .map((m) => {
      const overlap = m.genres.filter((g) => movie.genres.includes(g)).length;
      return { movie: m, overlap };
    })
    .filter((entry) => entry.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap || b.movie.rating - a.movie.rating)
    .slice(0, 14)
    .map(({ movie: m, overlap }) => ({
      ...m,
      score: scoreOf(m, 0.55 + overlap * 0.06),
    }));
}

/** GET /trending */
export async function getTrending(): Promise<ScoredMovie[]> {
  const { popular } = await loadCatalog();
  return popular.slice(0, 20);
}

/** GET /popular */
export async function getPopular(): Promise<ScoredMovie[]> {
  const { topRated } = await loadCatalog();
  return topRated.slice(0, 20);
}

/** GET /movies?genre= — a genre rail for the home page. */
export async function getByGenre(genre: string, limit = 20): Promise<ScoredMovie[]> {
  const { popular } = await loadCatalog();
  return popular.filter((m) => m.genres.includes(genre)).slice(0, limit);
}
