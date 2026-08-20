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
