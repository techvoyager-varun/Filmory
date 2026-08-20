import { memo } from "react";
import { Link } from "@tanstack/react-router";
import { X } from "lucide-react";
import { LikeButton, MyListButton, PlayButton } from "@/components/MovieActions";
import { Poster } from "@/components/Poster";
import { MatchBadge, RatingBadge } from "@/components/RatingBadge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ScoredMovie } from "@/types/movie";

export interface MovieCardProps {
  movie: ScoredMovie;
  rank?: number | undefined;
  progress?: number | undefined;
  subtitle?: string | undefined;
  onRemove?: (() => void) | undefined;
  className?: string | undefined;
}

function MovieCardBase({ movie, rank, progress, subtitle, onRemove, className }: MovieCardProps) {
  return (
    <article
      className={cn(
        "group/card relative w-[150px] shrink-0 sm:w-[170px] lg:w-[190px]",
        rank ? "pl-7" : "",
        className,
      )}
    >

      {rank ? (
        <span
          aria-hidden
          className="pointer-events-none absolute -bottom-1 left-0 font-display text-6xl font-extrabold leading-none text-surface-raised"
        >
          {rank}
        </span>
      ) : null}

      <Link
        to="/movies/$movieId"
        params={{ movieId: String(movie.movieId) }}
        className="block rounded-none outline-none ring-primary transition-transform duration-300 focus-visible:ring-2 group-hover/card:-translate-y-1.5 group-hover/card:scale-[1.04]"
      >
        <div className="relative overflow-hidden rounded-none bg-surface-raised shadow-lg shadow-background/60 ring-1 ring-border transition duration-300 group-hover/card:shadow-2xl group-hover/card:ring-gold/40">
          <Poster
            movie={movie}
            className="brightness-90 transition duration-500 group-hover/card:scale-[1.03] group-hover/card:brightness-110"
          />

          {/* Subtle tint only — keeps artwork in its original colours. */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 bg-gradient-to-t from-background/60 via-transparent to-transparent"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 bg-gradient-to-t from-background/90 via-background/10 to-transparent"
          />

          <div className="absolute inset-x-0 top-0 flex items-start justify-between gap-1 p-2">
            {typeof movie.score === "number" ? <MatchBadge score={movie.score} /> : <span />}
            <RatingBadge rating={movie.rating} />
          </div>


          <div className="absolute inset-x-0 bottom-0 translate-y-2 bg-gradient-to-t from-background via-background/85 to-transparent p-2 pt-8 opacity-0 transition duration-300 group-hover/card:translate-y-0 group-hover/card:opacity-100">
            <div className="flex items-center gap-1.5">
              <PlayButton movie={movie} size="icon" />
              <LikeButton movie={movie} size="icon" />
              <MyListButton movie={movie} size="icon" />
            </div>
          </div>

          {typeof progress === "number" ? (
            <div className="absolute inset-x-2 bottom-2 h-1 overflow-hidden rounded-none bg-foreground/25">
              <div className="h-full rounded-none bg-primary" style={{ width: `${progress}%` }} />
            </div>
          ) : null}
        </div>

        <div className="mt-2 space-y-0.5">
          <h3 className="truncate text-sm font-semibold">{movie.title}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {subtitle ??
              ([movie.year || null, movie.genres.slice(0, 2).join(" • ") || null]
                .filter(Boolean)
                .join(" • ") ||
                "Catalog title")}
          </p>
        </div>
      </Link>

      {onRemove ? (
        <Button
          size="icon"
          variant="secondary"
          aria-label={`Remove ${movie.title}`}
          onClick={onRemove}
          className="absolute right-2 top-2 size-7 rounded-none bg-background/80 opacity-0 transition group-hover/card:opacity-100 focus-visible:opacity-100"
        >
          <X className="size-3.5" aria-hidden />
        </Button>
      ) : null}
    </article>
  );
}

export const MovieCard = memo(MovieCardBase);
