/**
 * Single seam between the UI and the data source.
 *
 * Today every api/* function resolves mock data through `mock()`.
 * When the FastAPI backend exists, set VITE_API_BASE_URL and swap each
 * `mock(...)` call for `request(...)` — nothing else in the app changes.
 */

export const API_BASE_URL = import.meta.env["VITE_API_BASE_URL"] ?? "";

/** Simulated network latency so loading states are exercised in the prototype. */
export function mock<T>(data: T, delay = 320): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(data), delay));
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${path}`);
  }

  return (await response.json()) as T;
}
