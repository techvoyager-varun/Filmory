import { useEffect, useState } from "react";
import { Link } from "@tanstack/react-router";
import { Info } from "lucide-react";
import { LikeButton, MyListButton, PlayButton } from "@/components/MovieActions";
import { Backdrop, Poster } from "@/components/Poster";
import { RatingBadge } from "@/components/RatingBadge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ScoredMovie } from "@/types/movie";

const ROTATE_MS = 7000;

export function HeroBanner({ movies }: { movies: ScoredMovie[] }) {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused || movies.length < 2) return;
    const id = setInterval(() => setIndex((i) => (i + 1) % movies.length), ROTATE_MS);
    return () => clearInterval(id);
  }, [paused, movies.length]);

  const movie = movies[index];
  if (!movie) return null;

  return (
    <section
      className="relative h-[58vh] min-h-[420px] w-full overflow-hidden md:h-[74vh]"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      aria-roledescription="carousel"
      aria-label="Featured movies"
    >
      {movies.map((m, i) => (
        <div
          key={m.movieId}
          className={cn(
            "absolute inset-0 transition-opacity duration-1000 ease-out",
            i === index ? "opacity-95" : "opacity-0",
          )}
        >
          <Backdrop
            movie={m}
            className="object-top lg:scale-110 lg:object-center lg:blur-3xl lg:saturate-200"
          />
        </div>
      ))}


      <div
        key={movie.movieId}
        className="absolute right-6 top-1/2 hidden w-[19rem] -translate-y-1/2 animate-fade-in overflow-hidden rounded-lg border border-gold/25 shadow-[0_30px_80px_rgba(0,0,0,0.7)] lg:block"
      >
        <Poster movie={movie} eager />
      </div>

      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-background/40 via-transparent to-transparent"
      />

      <div className="hero-scrim absolute inset-0" />
      <div aria-hidden className="hero-vignette pointer-events-none absolute inset-0" />

      <div className="relative z-10 mx-auto flex h-full max-w-[1600px] flex-col justify-end px-4 pb-8 md:px-10 md:pb-14">
        <div key={movie.movieId} className="max-w-2xl animate-fade-in">
          <span className="inline-flex items-center gap-3 text-[10px] font-semibold uppercase tracking-[0.24em] text-gold md:text-[11px]">
            <span aria-hidden className="h-px w-6 bg-gold/70 md:w-8" />
            Featured
          </span>


          <h1 className="mt-3 text-2xl font-extrabold leading-[1.15] drop-shadow-[0_6px_30px_rgba(0,0,0,0.65)] sm:text-3xl md:mt-4 md:text-4xl lg:text-4xl xl:text-5xl max-w-xl">
            {movie.title}
          </h1>

          <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-xs text-muted-foreground md:mt-4 md:gap-x-3 md:gap-y-2 md:text-sm">
            <RatingBadge rating={movie.rating} />
            <span className="font-medium text-foreground/80">{movie.year}</span>
            <span aria-hidden className="text-gold/50">•</span>
            <span>{movie.genres.join(" • ")}</span>
            {movie.runtime ? (
              <>
                <span aria-hidden className="text-gold/50">•</span>
                <span>
                  {Math.floor(movie.runtime / 60)}h {movie.runtime % 60}m
                </span>
              </>
            ) : null}
          </div>

          <p className="mt-2.5 line-clamp-2 max-w-xl text-xs leading-relaxed text-muted-foreground md:mt-4 md:line-clamp-3 md:text-base">
            {movie.description}
          </p>


          <div className="mt-5 flex flex-wrap items-center gap-2 md:mt-7 md:gap-3">

            <PlayButton movie={movie} />
            <Button
              asChild
              size="lg"
              variant="secondary"
              className="h-10 rounded-md border border-border bg-surface-raised/80 px-4 text-sm hover:bg-surface-raised md:h-11 md:px-6 md:text-base"
            >
              <Link to="/movies/$movieId" params={{ movieId: String(movie.movieId) }}>
                <Info className="size-4" aria-hidden />
                More Info
              </Link>
            </Button>
            <LikeButton movie={movie} />
            <MyListButton movie={movie} />
          </div>
        </div>

        <div className="mt-6 flex gap-2 md:mt-8">
          {movies.map((m, i) => (
            <button
              key={m.movieId}
              type="button"
              aria-label={`Show ${m.title}`}
              aria-current={i === index}
              onClick={() => setIndex(i)}
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                i === index ? "w-8 bg-primary" : "w-3 bg-foreground/30 hover:bg-foreground/60",
              )}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
