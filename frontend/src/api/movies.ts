import { loadCatalog } from "@/data/catalog";
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

/** GET /genres — genre facets derived from the catalog itself. */
export async function getGenres(): Promise<string[]> {
  const { genres } = await loadCatalog();
  return genres;
}

/** GET /movies?genre=&sort=&offset=&limit= */
export async function browseMovies({
  genre = "All",
  sort = "popular",
  query = "",
  offset = 0,
  limit = 48,
}: BrowseParams = {}): Promise<MoviePage> {
  const { movies } = await loadCatalog();
  const q = query.trim().toLowerCase();

  let list = movies;
  if (genre !== "All") list = list.filter((m) => m.genres.includes(genre));
  if (q) list = list.filter((m) => m.title.toLowerCase().includes(q));

  const sorted = [...list].sort((a, b) => {
    if (sort === "rating") return b.rating - a.rating || b.ratingCount - a.ratingCount;
    if (sort === "year") return b.year - a.year || b.ratingCount - a.ratingCount;
    if (sort === "title") return a.title.localeCompare(b.title);
    return b.ratingCount - a.ratingCount;
  });

  return { movies: sorted.slice(offset, offset + limit), total: sorted.length };
}

/** GET /movies — a browsable slice of the catalog. */
export async function getMovies(): Promise<Movie[]> {
  const { popular } = await loadCatalog();
  return popular.slice(0, 60);
}

/** GET /movies/:id */
export async function getMovie(movieId: number): Promise<Movie | null> {
  const { byId } = await loadCatalog();
  return byId.get(movieId) ?? null;
}

/** GET /search?query= */
export async function searchMovies(query: string): Promise<Movie[]> {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const { movies } = await loadCatalog();
  const scored: { movie: Movie; weight: number }[] = [];

  for (const movie of movies) {
    const title = movie.title.toLowerCase();
    let weight = -1;
    if (title === q) weight = 4;
    else if (title.startsWith(q)) weight = 3;
    else if (title.includes(q)) weight = 2;
    else if (movie.genres.some((g) => g.toLowerCase() === q)) weight = 1;
    else if (String(movie.year) === q) weight = 1;
    if (weight >= 0) scored.push({ movie, weight });
  }

  return scored
    .sort(
      (a, b) =>
        b.weight - a.weight ||
        (b.movie as { ratingCount?: number }).ratingCount! -
          (a.movie as { ratingCount?: number }).ratingCount!,
    )
    .slice(0, 60)
    .map((s) => s.movie);
}

export async function getMoviesByIds(ids: number[]): Promise<Movie[]> {
  const { byId } = await loadCatalog();
  return ids.flatMap((id) => {
    const movie = byId.get(id);
    return movie ? [movie] : [];
  });
}
