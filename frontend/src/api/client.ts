/**
 * Single seam between the UI and the FastAPI backend.
 */

export const API_BASE_URL = import.meta.env["VITE_API_BASE_URL"] ?? "http://localhost:8000";

const TOKEN_KEY = "filmory.token";
export const AUTH_EXPIRED_EVENT = "filmory:auth-expired";
let authRevision = 0;
export const getAuthRevision = () => authRevision;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  authRevision += 1;
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  authRevision += 1;
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAuthToken();
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const url = path.startsWith("http")
    ? path
    : `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;

  const response = await fetch(url, {
    ...init,
    headers,
  });

  if (!response.ok) {
    if (response.status === 401 && token && token === getAuthToken()) {
      clearAuthToken();
      window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    }
    let errorDetail = `Request failed (${response.status}): ${path}`;
    try {
      const errorJson = await response.json();
      if (errorJson && errorJson.detail) {
        errorDetail =
          typeof errorJson.detail === "string"
            ? errorJson.detail
            : Array.isArray(errorJson.detail)
              ? errorJson.detail
                  .map((item: { msg?: string }) => item.msg ?? "Invalid input")
                  .join("; ")
              : "The server could not process this request.";
      }
    } catch {
      // ignore json parse error on non-json response
    }
    throw new ApiError(errorDetail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
