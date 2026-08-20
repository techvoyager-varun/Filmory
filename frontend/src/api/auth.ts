import { mock } from "./client";
import { store } from "./mock-store";
import type { AuthCredentials, RegisterPayload, User } from "@/types/movie";

/** MOCK auth only — no real credential checking until the backend exists. */

const DEMO_USER: User = {
  userId: "demo-user",
  name: "Demo Viewer",
  email: "demo@filmory.app",
  favoriteGenres: ["Sci-Fi", "Drama", "Thriller"],
  favoriteMovieIds: [109487, 79132, 152081],
};

/** POST /auth/login */
export function login({ email }: AuthCredentials): Promise<User> {
  const user: User = {
    userId: `user-${email.split("@")[0] || "guest"}`,
    name: (email.split("@")[0] || "Viewer").replace(/^\w/, (c) => c.toUpperCase()),
    email,
    favoriteGenres: [],
    favoriteMovieIds: [],
  };
  store.setUser(user);
  return mock(user, 500);
}

/** POST /auth/register */
export function register({ name, email }: RegisterPayload): Promise<User> {
  const user: User = {
    userId: `user-${Date.now()}`,
    name,
    email,
    favoriteGenres: [],
    favoriteMovieIds: [],
  };
  store.setUser(user);
  return mock(user, 550);
}

export function loginAsDemo(): Promise<User> {
  store.setUser(DEMO_USER);
  return mock(DEMO_USER, 300);
}

export function getCurrentUser(): User | null {
  return store.getUser();
}

export function updatePreferences(genres: string[], movieIds: number[]): Promise<User | null> {
  const user = store.getUser();
  if (!user) return mock(null, 100);
  const next: User = { ...user, favoriteGenres: genres, favoriteMovieIds: movieIds };
  store.setUser(next);
  return mock(next, 300);
}

export function logout(): void {
  store.setUser(null);
}
