import { request } from "./client";
import type {
  ChatRequest,
  ChatResponse,
  FeedbackRequest,
  MovieIntent,
  SessionDetail,
  SessionSummary,
} from "../types/assistant";

export async function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/assistant/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/api/assistant/sessions");
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/assistant/sessions/${sessionId}`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request<void>(`/api/assistant/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export async function submitFeedback(payload: FeedbackRequest): Promise<{ status: string }> {
  return request<{ status: string }>("/api/assistant/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function clearSessionFilter(sessionId: string, filterKey: string): Promise<MovieIntent> {
  return request<MovieIntent>(`/api/assistant/sessions/${sessionId}/filters/${encodeURIComponent(filterKey)}`, {
    method: "DELETE",
  });
}

