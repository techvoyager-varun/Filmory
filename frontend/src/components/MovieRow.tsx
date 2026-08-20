import { useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { MovieCard } from "@/components/MovieCard";
import { RowSkeleton } from "@/components/LoadingSkeleton";
import { ErrorState } from "@/components/ErrorState";
import { Button } from "@/components/ui/button";
import type { ScoredMovie } from "@/types/movie";

export interface MovieRowProps {
  title: string;
  subtitle?: string;
  movies: ScoredMovie[];
  loading?: boolean;
  error?: boolean;
  onRetry?: (() => void) | undefined;
  ranked?: boolean;
  progressFor?: (movie: ScoredMovie) => number | undefined;
  subtitleFor?: (movie: ScoredMovie) => string | undefined;
}

export function MovieRow({
  title,
  subtitle,
  movies,
  loading,
  error,
  onRetry,
  ranked,
  progressFor,
  subtitleFor,
}: MovieRowProps) {
  const scroller = useRef<HTMLDivElement>(null);

  const scrollBy = (direction: 1 | -1) => {
    const el = scroller.current;
    if (!el) return;
    el.scrollBy({ left: direction * Math.max(320, el.clientWidth * 0.8), behavior: "smooth" });
  };

  if (!loading && !error && movies.length === 0) return null;

  return (
    <section className="content-auto group/row relative py-6" aria-label={title}>
      <div className="mb-3 flex items-end justify-between gap-4 px-4 md:px-10">
        <div className="border-l-2 border-gold/70 pl-3">
          <h2 className="text-lg font-bold tracking-tight md:text-xl">{title}</h2>
          {subtitle ? <p className="text-xs text-muted-foreground">{subtitle}</p> : null}
        </div>
      </div>

      {error ? (
        <div className="px-4 md:px-10">
          <ErrorState
            compact
            title={`Couldn't load ${title}`}
            description="This row will populate once the recommendation service responds."
            onRetry={onRetry}
          />
        </div>
      ) : loading ? (
        <RowSkeleton />
      ) : (
        <div className="relative">
          <div
            ref={scroller}
            role="group"
            aria-label={`${title} carousel — use left and right arrow keys to scroll`}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowRight") {
                e.preventDefault();
                scrollBy(1);
              } else if (e.key === "ArrowLeft") {
                e.preventDefault();
                scrollBy(-1);
              }
            }}
            className="scrollbar-none flex gap-3 overflow-x-auto scroll-smooth px-4 pb-2 pt-1 outline-none focus-visible:ring-2 focus-visible:ring-primary md:gap-4 md:px-10"
          >
            {movies.map((movie, index) => (
              <div key={movie.movieId}>
                <MovieCard
                  movie={movie}
                  rank={ranked ? index + 1 : undefined}
                  progress={progressFor?.(movie)}
                  subtitle={subtitleFor?.(movie)}
                />
              </div>
            ))}
          </div>

          <div className="row-fade-left pointer-events-none absolute inset-y-0 left-0 hidden w-10 md:block" />
          <div className="row-fade-right pointer-events-none absolute inset-y-0 right-0 hidden w-14 md:block" />

          <Button
            size="icon"
            variant="secondary"
            aria-label={`Scroll ${title} left`}
            onClick={() => scrollBy(-1)}
            className="absolute left-1 top-[38%] hidden size-10 -translate-y-1/2 rounded-none border border-border bg-background/70 opacity-0 backdrop-blur transition hover:bg-background focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-primary group-hover/row:opacity-100 md:flex"
          >
            <ChevronLeft className="size-5" aria-hidden />
          </Button>
          <Button
            size="icon"
            variant="secondary"
            aria-label={`Scroll ${title} right`}
            onClick={() => scrollBy(1)}
            className="absolute right-1 top-[38%] hidden size-10 -translate-y-1/2 rounded-none border border-border bg-background/70 opacity-0 backdrop-blur transition hover:bg-background focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-primary group-hover/row:opacity-100 md:flex"
          >
            <ChevronRight className="size-5" aria-hidden />
          </Button>
        </div>
      )}

    </section>
  );
}
