import { CURATED, cdn } from "@/data/movies";
import type { Movie } from "@/types/movie";

/**
 * The real MovieLens catalog (27k+ titles) shipped as a static JSON file and
 * loaded once on demand. When the FastAPI backend exists these lookups move
 * behind HTTP endpoints — the shapes returned here already match.
 */

interface CatalogFile {
  genres: string[];
  /** [movieId, title, genreIndexes, year, rating(0-5), ratingCount] */
  movies: [number, string, number[], number, number, number][];
}

export interface CatalogMovie extends Movie {
  ratingCount: number;
}

export interface Catalog {
  /** Genre names present in the catalog file (excludes the "no genres" marker). */
  genres: string[];
  movies: CatalogMovie[];
  byId: Map<number, CatalogMovie>;
  /** Sorted by popularity (number of ratings). */
  popular: CatalogMovie[];
  /** Highly rated titles with enough ratings to be meaningful. */
  topRated: CatalogMovie[];
  /** Curated titles that have official artwork — used for hero/feature slots. */
  featured: CatalogMovie[];
}

const TITLE_ARTICLE = /,\s(The|A|An|Les|La|Le|Il|El|Der|Die|Das)$/;

function cleanTitle(raw: string): { title: string; year: number } {
  const match = raw.match(/^(.*)\s\((\d{4})\)\s*$/);
  let title = match ? match[1]! : raw;
  const year = match ? Number(match[2]) : 0;

  const article = title.match(TITLE_ARTICLE);
  if (article) title = `${article[1]} ${title.replace(TITLE_ARTICLE, "")}`;

  return { title: title.trim(), year };
}

function synopsis(title: string, genres: string[], year: number, ratingCount: number): string {
  const list = genres.filter((g) => g !== "(no genres listed)");
  const genreText = list.length ? list.join(", ").toLowerCase() : "uncategorised";
  return `${title}${year ? ` (${year})` : ""} is a ${genreText} title from the MovieLens catalog, rated by ${ratingCount.toLocaleString()} viewers.`;
}

let cached: Promise<Catalog> | null = null;

async function build(): Promise<Catalog> {
  const response = await fetch("/movie-catalog.json");
  if (!response.ok) throw new Error(`Catalog request failed (${response.status})`);
  const file = (await response.json()) as CatalogFile;

  const movies: CatalogMovie[] = file.movies.map(([movieId, rawTitle, genreIds, year, rating, ratingCount]) => {
    const { title, year: parsedYear } = cleanTitle(rawTitle);
    const genres = genreIds
      .map((i) => file.genres[i]!)
      .filter((g) => Boolean(g) && g !== "(no genres listed)");
    const extra = CURATED[movieId];
    const poster = extra ? cdn(extra.poster, 1000) : "";

    return {
      movieId,
      title,
      genres,
      year: year || parsedYear,
      rating: Number((rating * 2).toFixed(1)),
      ratingCount,
      runtime: extra?.runtime ?? 0,
      posterUrl: poster,
      backdropUrl: extra ? cdn(extra.poster, 2400) : "",
      description:
        extra?.description ?? synopsis(title, genres, year || parsedYear, ratingCount),
    };
  });

  const byId = new Map(movies.map((m) => [m.movieId, m]));
  const popular = [...movies].sort((a, b) => b.ratingCount - a.ratingCount);
  const topRated = movies
    .filter((m) => m.ratingCount >= 5000)
    .sort((a, b) => b.rating - a.rating || b.ratingCount - a.ratingCount);
  const featured = Object.keys(CURATED)
    .map((id) => byId.get(Number(id)))
    .filter((m): m is CatalogMovie => Boolean(m))
    .sort((a, b) => b.rating - a.rating);

  const genres = file.genres.filter((g) => g !== "(no genres listed)");

  return { genres, movies, byId, popular, topRated, featured };
}

export function loadCatalog(): Promise<Catalog> {
  cached ??= build().catch((error) => {
    cached = null;
    throw error;
  });
  return cached;
}
