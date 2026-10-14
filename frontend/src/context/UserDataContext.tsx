import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
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
  hiddenMovies: number[];
  loading: boolean;
  isInList: (movieId: number) => boolean;
  isLiked: (movieId: number) => boolean;
  isHidden: (movieId: number) => boolean;
  toggleHidden: (movie: Movie) => Promise<void>;
  play: (movie: Movie) => Promise<void>;
  toggleLike: (movie: Movie) => Promise<void>;
  toggleMyList: (movie: Movie) => Promise<void>;
  removeFromList: (movie: Movie) => Promise<void>;
}

const UserDataContext = createContext<UserDataContextValue | null>(null);

export function UserDataProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const userId = user?.userId ?? "guest";
  const activeUserId = useRef(userId);
  activeUserId.current = userId;
  const queryClient = useQueryClient();
  const pending = useRef(new Set<string>());
  const runAction = useCallback(
    async (key: string, action: () => Promise<void>) => {
      if (!user) {
        toast.error("Please sign in to save movies or record activity.");
        return;
      }
      const requestKey = `${userId}:${key}`;
      if (pending.current.has(requestKey)) return;
      pending.current.add(requestKey);
      try {
        await action();
      } catch (error) {
        if (activeUserId.current === userId)
          toast.error(
            error instanceof Error ? error.message : "Could not save this change. Please retry.",
          );
      } finally {
        pending.current.delete(requestKey);
      }
    },
    [user, userId],
  );

  const [myList, setMyList] = useState<Movie[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [likes, setLikes] = useState<number[]>([]);
  const [hiddenMovies, setHiddenMovies] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setMyList([]);
    setHistory([]);
    setLikes([]);
    setHiddenMovies([]);
    setLoading(true);
    if (!user) {
      setMyList([]);
      setHistory([]);
      setLikes([]);
      setLoading(false);
      return;
    }
    try {
      const [list, hist, liked, hidden] = await Promise.all([
        interactionsApi.getMyList(userId),
        interactionsApi.getHistory(userId),
        interactionsApi.getLikes(),
        interactionsApi.getHiddenMovies(),
      ]);
      if (activeUserId.current !== userId) return;
      setMyList(list);
      setHistory(hist);
      setLikes(liked);
      setHiddenMovies(hidden);
    } catch (error) {
      if (activeUserId.current === userId)
        toast.error(
          error instanceof Error
            ? error.message
            : "Could not load your saved movies. Please retry.",
        );
    } finally {
      if (activeUserId.current === userId) setLoading(false);
    }
  }, [user, userId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const play = useCallback(
    async (movie: Movie) => {
      await interactionsApi.recordInteraction({ userId, movieId: movie.movieId, type: "play" });
      if (activeUserId.current !== userId) return;
      const next = await interactionsApi.getHistory(userId);
      if (activeUserId.current !== userId) return;
      setHistory(next);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
      void queryClient.invalidateQueries({ queryKey: ["taste-state"] });
      toast.success(`Now playing — ${movie.title}`, {
        description:
          "Playback is simulated. The interaction was recorded for your recommendations.",
      });
    },
    [userId, queryClient],
  );

  const toggleLike = useCallback(
    async (movie: Movie) => {
      const wasLiked = likes.includes(movie.movieId);
      const next = await interactionsApi.toggleLike(userId, movie.movieId);
      if (activeUserId.current !== userId) return;
      setLikes(next);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
      void queryClient.invalidateQueries({ queryKey: ["taste-state"] });
      toast.success(wasLiked ? `Removed like from ${movie.title}` : `Liked ${movie.title}`);
    },
    [likes, userId, queryClient],
  );

  const toggleMyList = useCallback(
    async (movie: Movie) => {
      const inList = myList.some((m) => m.movieId === movie.movieId);
      const next = inList
        ? await interactionsApi.removeFromMyList(userId, movie.movieId)
        : await interactionsApi.addToMyList(userId, movie.movieId);
      if (activeUserId.current !== userId) return;
      setMyList(next);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
      toast.success(
        inList ? `Removed ${movie.title} from My List` : `Added ${movie.title} to My List`,
      );
    },
    [myList, userId, queryClient],
  );

  const removeFromList = useCallback(
    async (movie: Movie) => {
      const next = await interactionsApi.removeFromMyList(userId, movie.movieId);
      if (activeUserId.current !== userId) return;
      setMyList(next);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
      toast.success(`Removed ${movie.title} from My List`);
    },
    [userId, queryClient],
  );

  const toggleHidden = useCallback(
    async (movie: Movie) => {
      const wasHidden = hiddenMovies.includes(movie.movieId);
      await interactionsApi.recordInteraction({
        movieId: movie.movieId,
        type: wasHidden ? "show_again" : "not_interested",
      });
      const [hidden, liked] = await Promise.all([
        interactionsApi.getHiddenMovies(),
        interactionsApi.getLikes(),
      ]);
      if (activeUserId.current !== userId) return;
      setHiddenMovies(hidden);
      setLikes(liked);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
      void queryClient.invalidateQueries({ queryKey: ["taste-state"] });
      toast.success(
        wasHidden
          ? `${movie.title} can be recommended again`
          : `Hidden ${movie.title} from recommendations`,
        {
          description: wasHidden
            ? undefined
            : "You can undo this from the movie page or your profile.",
        },
      );
    },
    [hiddenMovies, userId, queryClient],
  );

  const value = useMemo(
    () => ({
      myList,
      history,
      likes,
      hiddenMovies,
      loading,
      isInList: (movieId: number) => myList.some((m) => m.movieId === movieId),
      isLiked: (movieId: number) => likes.includes(movieId),
      isHidden: (movieId: number) => hiddenMovies.includes(movieId),
      toggleHidden: (movie: Movie) =>
        runAction(`hidden:${movie.movieId}`, () => toggleHidden(movie)),
      play: (movie: Movie) => runAction(`play:${movie.movieId}`, () => play(movie)),
      toggleLike: (movie: Movie) => runAction(`like:${movie.movieId}`, () => toggleLike(movie)),
      toggleMyList: (movie: Movie) => runAction(`list:${movie.movieId}`, () => toggleMyList(movie)),
      removeFromList: (movie: Movie) =>
        runAction(`list:${movie.movieId}`, () => removeFromList(movie)),
    }),
    [
      myList,
      history,
      likes,
      hiddenMovies,
      loading,
      play,
      toggleLike,
      toggleMyList,
      toggleHidden,
      removeFromList,
      runAction,
    ],
  );

  return <UserDataContext.Provider value={value}>{children}</UserDataContext.Provider>;
}

export function useUserData() {
  const ctx = useContext(UserDataContext);
  if (!ctx) throw new Error("useUserData must be used inside <UserDataProvider>");
  return ctx;
}
