import type { Interaction, User } from "@/types/movie";

/**
 * Local persistence used ONLY while the app runs on mock data.
 * Once the FastAPI backend is wired up this module is deleted and the
 * api/* functions call the real endpoints instead.
 */

const KEYS = {
  user: "filmory.user",
  interactions: "filmory.interactions",
  myList: "filmory.mylist",
  likes: "filmory.likes",
};

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — prototype keeps working in memory */
  }
}

export const store = {
  getUser: () => read<User | null>(KEYS.user, null),
  setUser: (user: User | null) =>
    user ? write(KEYS.user, user) : window?.localStorage?.removeItem(KEYS.user),

  getInteractions: () => read<Interaction[]>(KEYS.interactions, []),
  addInteraction: (interaction: Interaction) => {
    const all = read<Interaction[]>(KEYS.interactions, []);
    all.unshift(interaction);
    write(KEYS.interactions, all.slice(0, 200));
  },

  getMyList: () => read<number[]>(KEYS.myList, []),
  setMyList: (ids: number[]) => write(KEYS.myList, ids),

  getLikes: () => read<number[]>(KEYS.likes, []),
  setLikes: (ids: number[]) => write(KEYS.likes, ids),
};
