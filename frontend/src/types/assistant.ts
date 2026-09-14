import type { ScoredMovie } from "./movie";

export type RequestMode = "catalog_lookup" | "personalized_recommendation" | "similar_movies";

export interface MovieIntent {
  request_mode?: RequestMode | undefined;
  semantic_query: string;
  preferred_genres: string[];
  excluded_genres: string[];
  min_year: number | null;
  max_year: number | null;
  max_runtime_minutes: number | null;
  min_rating: number | null;
  reference_titles: string[];
  excluded_movie_ids: number[];
  mood: string;
  needs_clarification: boolean;
  clarification_question: string;
}

export interface MovieEvidence {
  movie_id: number;
  title: string;
  genres: string[];
  year?: number | null | undefined;
  runtime?: number | null | undefined;
  rating?: number | null | undefined;
  description_snippet: string;
  matched_constraints: string[];
  personalization_signals: string[];
  score: number;
  damr_variant: string;
  evidence_keys?: string[] | undefined;
}

export interface ExplanationItem {
  movie_id: number;
  explanation: string;
  evidence_ids: string[];
}

export interface StructuredExplanation {
  summary: string;
  items: ExplanationItem[];
}

export interface ChatRequest {
  message: string;
  session_id?: string | null | undefined;
}

export interface ChatResponse {
  session_id: string;
  message: string;
  movies: ScoredMovie[];
  intent: MovieIntent;
  evidence: MovieEvidence[];
  structured_explanation?: StructuredExplanation | null | undefined;
  clarification?: string | null | undefined;
  notice?: string | null | undefined;
  request_mode?: RequestMode | undefined;
}

export interface MessageSchema {
  role: "user" | "assistant";
  content: string;
  movies?: ScoredMovie[] | undefined;
  intent?: MovieIntent | null | undefined;
  evidence?: MovieEvidence[] | undefined;
  structured_explanation?: StructuredExplanation | null | undefined;
  notice?: string | null | undefined;
  request_mode?: RequestMode | undefined;
  timestamp: string;
}

export interface SessionSummary {
  session_id: string;
  preview: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail {
  session_id: string;
  messages: MessageSchema[];
  active_intent: MovieIntent;
  created_at: string;
  updated_at: string;
}

export interface FeedbackRequest {
  session_id: string;
  message_id?: number | null | undefined;
  movie_id?: number | null | undefined;
  feedback_type: "helpful" | "not_helpful" | "wrong_movie" | "bad_explanation" | "already_watched";
  comment?: string | null | undefined;
}
