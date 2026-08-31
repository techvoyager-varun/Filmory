import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { toast } from "sonner";
import * as interactionsApi from "@/api/interactions";
import type { HistoryEntry } from "@/api/interactions";
import { useAuth } from "@/context/AuthContext";
import type { Movie } from "@/types/movie";

interface UserDataContextValue {
  myList: Movie[];
  history: HistoryEntry[];
  likes: number[];
  loading: boolean;
  isInList: (movieId: number) => boolean;
  isLiked: (movieId: number) => boolean;
  play: (movie: Movie) => Promise<void>;
  toggleLike: (movie: Movie) => Promise<void>;
  toggleMyList: (movie: Movie) => Promise<void>;
  removeFromList: (movie: Movie) => Promise<void>;
}

const UserDataContext = createContext<UserDataContextValue | null>(null);

export function UserDataProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const userId = user?.userId ?? "guest";

  const [myList, setMyList] = useState<Movie[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [likes, setLikes] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!user) {
      setMyList([]);
      setHistory([]);
      setLikes([]);
      setLoading(false);
      return;
    }
    const [list, hist, liked] = await Promise.all([
      interactionsApi.getMyList(userId),
      interactionsApi.getHistory(userId),
      interactionsApi.getLikes(),
    ]);
    setMyList(list);
    setHistory(hist);
    setLikes(liked);
    setLoading(false);
  }, [user, userId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const play = useCallback(
    async (movie: Movie) => {
      await interactionsApi.recordInteraction({ userId, movieId: movie.movieId, type: "play" });
      setHistory(await interactionsApi.getHistory(userId));
      toast.success(`Now playing — ${movie.title}`, {
        description: "Playback is simulated. The interaction was recorded for your recommendations.",
      });
    },
    [userId],
  );

  const toggleLike = useCallback(
    async (movie: Movie) => {
      const wasLiked = likes.includes(movie.movieId);
      const next = await interactionsApi.toggleLike(userId, movie.movieId);
      setLikes(next);
      toast.success(wasLiked ? `Removed like from ${movie.title}` : `Liked ${movie.title}`);
    },
    [likes, userId],
  );

  const toggleMyList = useCallback(
    async (movie: Movie) => {
      const inList = myList.some((m) => m.movieId === movie.movieId);
      const next = inList
        ? await interactionsApi.removeFromMyList(userId, movie.movieId)
        : await interactionsApi.addToMyList(userId, movie.movieId);
      setMyList(next);
      toast.success(inList ? `Removed ${movie.title} from My List` : `Added ${movie.title} to My List`);
    },
    [myList, userId],
  );

  const removeFromList = useCallback(
    async (movie: Movie) => {
      setMyList(await interactionsApi.removeFromMyList(userId, movie.movieId));
      toast.success(`Removed ${movie.title} from My List`);
    },
    [userId],
  );

  const value = useMemo(
    () => ({
      myList,
      history,
      likes,
      loading,
      isInList: (movieId: number) => myList.some((m) => m.movieId === movieId),
      isLiked: (movieId: number) => likes.includes(movieId),
      play,
      toggleLike,
      toggleMyList,
      removeFromList,
    }),
    [myList, history, likes, loading, play, toggleLike, toggleMyList, removeFromList],
  );

  return <UserDataContext.Provider value={value}>{children}</UserDataContext.Provider>;
}

export function useUserData() {
  const ctx = useContext(UserDataContext);
  if (!ctx) throw new Error("useUserData must be used inside <UserDataProvider>");
  return ctx;
}
