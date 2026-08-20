import { mock } from "./client";
import { store } from "./mock-store";
import { loadCatalog } from "@/data/catalog";
import type { Interaction, InteractionType, Movie } from "@/types/movie";

export interface HistoryEntry {
  movie: Movie;
  watchedAt: string;
  progress: number;
}

/** POST /interactions */
export function recordInteraction(data: {
  userId: string;
  movieId: number;
  type: InteractionType;
}): Promise<Interaction> {
  const interaction: Interaction = { ...data, timestamp: new Date().toISOString() };
  store.addInteraction(interaction);
  // eslint-disable-next-line no-console
  console.info("[interaction]", interaction);
  return mock(interaction, 120);
}

/** GET /users/:id/history */
export async function getHistory(_userId: string): Promise<HistoryEntry[]> {
  const { byId } = await loadCatalog();
  const seen = new Set<number>();
  const entries: HistoryEntry[] = [];

  for (const i of store.getInteractions()) {
    if (i.type !== "play" || seen.has(i.movieId)) continue;
    const movie = byId.get(i.movieId);
    if (!movie) continue;
    seen.add(i.movieId);
    entries.push({
      movie,
      watchedAt: i.timestamp,
      progress: Math.min(95, 15 + ((i.movieId % 8) * 11)),
    });
  }

  return mock(entries, 260);
}

/** GET /users/:id/my-list */
export async function getMyList(_userId: string): Promise<Movie[]> {
  const { byId } = await loadCatalog();
  const ids = store.getMyList();
  return ids.flatMap((id) => {
    const movie = byId.get(id);
    return movie ? [movie] : [];
  });
}

/** POST /my-list */
export function addToMyList(userId: string, movieId: number): Promise<Movie[]> {
  const ids = store.getMyList();
  if (!ids.includes(movieId)) store.setMyList([movieId, ...ids]);
  void recordInteraction({ userId, movieId, type: "list_add" });
  return getMyList(userId);
}

/** DELETE /my-list/:movieId */
export function removeFromMyList(userId: string, movieId: number): Promise<Movie[]> {
  store.setMyList(store.getMyList().filter((id) => id !== movieId));
  void recordInteraction({ userId, movieId, type: "list_remove" });
  return getMyList(userId);
}

export function getLikes(): Promise<number[]> {
  return mock(store.getLikes(), 120);
}

export function toggleLike(userId: string, movieId: number): Promise<number[]> {
  const likes = store.getLikes();
  const liked = likes.includes(movieId);
  const next = liked ? likes.filter((id) => id !== movieId) : [movieId, ...likes];
  store.setLikes(next);
  void recordInteraction({ userId, movieId, type: liked ? "unlike" : "like" });
  return mock(next, 120);
}
