import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { ThumbsUp, ThumbsDown, Eye, Check } from "lucide-react";
import { Poster } from "@/components/Poster";
import { RatingBadge } from "@/components/RatingBadge";
import { LikeButton, MyListButton } from "@/components/MovieActions";
import { submitFeedback } from "@/api/assistant";
import type { ScoredMovie } from "@/types/movie";
import { cn } from "@/lib/utils";

interface AssistantMovieCardsProps {
  movies: ScoredMovie[];
  sessionId: string;
  messageId?: number;
  className?: string;
}

export function AssistantMovieCards({
  movies,
  sessionId,
  messageId,
  className,
}: AssistantMovieCardsProps) {
  const [feedbackSent, setFeedbackSent] = useState<Record<number, string>>({});

  if (!movies || movies.length === 0) return null;

  const handleFeedback = async (movieId: number, type: "helpful" | "not_helpful" | "already_watched") => {
    try {
      await submitFeedback({
        session_id: sessionId,
        message_id: messageId,
        movie_id: movieId,
        feedback_type: type,
      });
      setFeedbackSent((prev) => ({ ...prev, [movieId]: type }));
    } catch (e) {
      console.error("Failed to submit feedback:", e);
    }
  };

  return (
    <div className={cn("mt-4 space-y-3", className)}>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3.5">
        {movies.map((movie) => {
          const sentType = feedbackSent[movie.movieId];
          return (
            <div
              key={movie.movieId}
              className="group relative flex flex-col overflow-hidden rounded-xl border border-border/70 bg-surface shadow-md transition-all duration-300 hover:border-gold/50 hover:shadow-xl hover:-translate-y-1"
            >
              {/* Poster Link */}
              <Link
                to="/movies/$movieId"
                params={{ movieId: String(movie.movieId) }}
                className="relative block aspect-[2/3] w-full overflow-hidden bg-surface-raised"
              >
                <Poster
                  movie={movie}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                <div className="absolute top-2 right-2">
                  <RatingBadge rating={movie.rating} />
                </div>
              </Link>

              {/* Movie Info */}
              <div className="flex flex-1 flex-col justify-between p-2.5">
                <div>
                  <Link
                    to="/movies/$movieId"
                    params={{ movieId: String(movie.movieId) }}
                    className="block font-medium text-xs text-foreground line-clamp-1 hover:text-gold transition-colors"
                  >
                    {movie.title}
                  </Link>
                  <div className="mt-0.5 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>{movie.year}</span>
                    {movie.score ? (
                      <span className="font-mono text-gold text-[10px]">
                        {(movie.score * 100).toFixed(0)}% match
                      </span>
                    ) : null}
                  </div>
                  {movie.genres && movie.genres.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {movie.genres.slice(0, 2).map((g) => (
                        <span
                          key={g}
                          className="rounded bg-surface-raised px-1.5 py-0.5 text-[9px] text-muted-foreground"
                        >
                          {g}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Card Actions & Feedback */}
                <div className="mt-2.5 pt-2 border-t border-border/50 flex items-center justify-between">
                  <div className="flex items-center gap-1">
                    <LikeButton movie={movie} size="icon" />
                    <MyListButton movie={movie} size="icon" />
                  </div>

                  {/* Feedback Mini Toolbar */}
                  <div className="flex items-center gap-1">
                    {sentType ? (
                      <span className="inline-flex items-center gap-0.5 text-[10px] text-emerald-400 font-medium">
                        <Check className="h-3 w-3" />
                        Feedback
                      </span>
                    ) : (
                      <>
                        <button
                          type="button"
                          title="Good recommendation"
                          onClick={() => handleFeedback(movie.movieId, "helpful")}
                          className="rounded p-1 text-muted-foreground/70 hover:bg-surface-raised hover:text-emerald-400 transition"
                        >
                          <ThumbsUp className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          title="Not what I wanted"
                          onClick={() => handleFeedback(movie.movieId, "not_helpful")}
                          className="rounded p-1 text-muted-foreground/70 hover:bg-surface-raised hover:text-red-400 transition"
                        >
                          <ThumbsDown className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          title="Already watched"
                          onClick={() => handleFeedback(movie.movieId, "already_watched")}
                          className="rounded p-1 text-muted-foreground/70 hover:bg-surface-raised hover:text-primary transition"
                        >
                          <Eye className="h-3 w-3" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
