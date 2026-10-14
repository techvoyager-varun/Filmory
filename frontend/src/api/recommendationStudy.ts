import { request } from "./client";
import type { ScoredMovie } from "@/types/movie";

export type StudyChoice = "interesting" | "not_for_me" | "unsure";
export interface RecommendationStudy {
  id: string;
  movies: ScoredMovie[];
  answers: Record<string, StudyChoice>;
  createdAt: string;
}

export function startRecommendationStudy(): Promise<RecommendationStudy> {
  return request("/api/recommendation-study", {
    method: "POST",
    body: JSON.stringify({ consent: true }),
  });
}

export function saveStudyAnswer(
  id: string,
  movieId: number,
  choice: StudyChoice,
): Promise<RecommendationStudy> {
  return request(`/api/recommendation-study/${id}/answers`, {
    method: "PUT",
    body: JSON.stringify({ movieId, choice }),
  });
}
