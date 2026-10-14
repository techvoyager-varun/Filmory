import { request, setAuthToken, clearAuthToken, getAuthToken, getAuthRevision } from "./client";
import type { AuthCredentials, RegisterPayload, User } from "@/types/movie";

const USER_KEY = "filmory.user";

interface AuthResponse {
  accessToken: string;
  tokenType: string;
  user: User;
}

function saveUser(user: User | null): void {
  if (typeof window === "undefined") return;
  if (user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(USER_KEY);
  }
}

/** POST /api/auth/login */
export async function login({ email, password }: AuthCredentials): Promise<User> {
  const revision = getAuthRevision();
  const previousToken = getAuthToken();
  const data = await request<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (revision !== getAuthRevision() || previousToken !== getAuthToken())
    throw new Error("Your session changed. Please sign in again.");
  setAuthToken(data.accessToken);
  saveUser(data.user);
  return data.user;
}

/** POST /api/auth/register */
export async function register({ name, email, password }: RegisterPayload): Promise<User> {
  const revision = getAuthRevision();
  const previousToken = getAuthToken();
  const data = await request<AuthResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ name, email, password }),
  });
  if (revision !== getAuthRevision() || previousToken !== getAuthToken())
    throw new Error("Your session changed. Please sign in again.");
  setAuthToken(data.accessToken);
  saveUser(data.user);
  return data.user;
}

/** POST /api/auth/demo */
export async function loginAsDemo(): Promise<User> {
  const revision = getAuthRevision();
  const previousToken = getAuthToken();
  const data = await request<AuthResponse>("/api/auth/demo", {
    method: "POST",
  });
  if (revision !== getAuthRevision() || previousToken !== getAuthToken())
    throw new Error("Your session changed. Please sign in again.");
  setAuthToken(data.accessToken);
  saveUser(data.user);
  return data.user;
}

export function getCurrentUser(): User | null {
  if (typeof window === "undefined") return null;
  if (!getAuthToken()) return null;
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  } catch {
    return null;
  }
}

export async function restoreSession(): Promise<User | null> {
  const token = getAuthToken();
  if (!token) return null;
  const user = await request<User>("/api/auth/me");
  if (token !== getAuthToken()) return null;
  saveUser(user);
  return user;
}

/** PUT /api/auth/preferences */
export async function updatePreferences(
  genres: string[],
  movieIds: number[],
): Promise<User | null> {
  const token = getAuthToken();
  const user = await request<User>("/api/auth/preferences", {
    method: "PUT",
    body: JSON.stringify({ favoriteGenres: genres, favoriteMovieIds: movieIds }),
  });
  if (token !== getAuthToken()) return null;
  saveUser(user);
  return user;
}

export function logout(): void {
  clearAuthToken();
  saveUser(null);
}
