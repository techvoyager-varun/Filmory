import { useState } from "react";
import { cn } from "@/lib/utils";
import type { Movie } from "@/types/movie";

/** Deterministic hue so a title always gets the same generated artwork. */
function hueOf(movie: Pick<Movie, "movieId" | "title">) {
  let hash = movie.movieId;
  for (const ch of movie.title) hash = (hash * 31 + ch.charCodeAt(0)) % 100000;
  return hash % 360;
}

function initials(title: string) {
  return title
    .split(/\s+/)
    .filter((w) => /[a-z0-9]/i.test(w))
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("");
}

/**
 * Poster artwork. Catalog titles without official artwork get a generated
 * typographic plate instead of a broken image.
 */
export function Poster({
  movie,
  className,
  eager,
}: {
  movie: Movie;
  className?: string;
  eager?: boolean;
}) {
  const [error, setError] = useState(false);

  if (movie.posterUrl && !error) {
    return (
      <img
        src={movie.posterUrl}
        alt={`${movie.title} poster`}
        loading={eager ? "eager" : "lazy"}
        decoding="async"
        width={500}
        height={750}
        onError={() => setError(true)}
        className={cn("aspect-[2/3] w-full object-cover", className)}
      />
    );
  }

  const hue = hueOf(movie);

  return (
    <div
      role="img"
      aria-label={`${movie.title} artwork`}
      className={cn(
        "relative flex aspect-[2/3] w-full flex-col justify-between overflow-hidden p-3",
        className,
      )}
      style={{
        background: `linear-gradient(160deg, hsl(${hue} 32% 18%) 0%, hsl(${(hue + 40) % 360} 28% 10%) 55%, hsl(0 0% 6%) 100%)`,
      }}
    >
      <span
        aria-hidden
        className="absolute -right-4 bottom-2 font-display text-[5.5rem] font-extrabold leading-none text-foreground/10"
      >
        {initials(movie.title)}
      </span>
      <span className="relative text-[10px] font-semibold uppercase tracking-[0.2em] text-gold/80">
        {movie.genres[0] ?? "Film"}
      </span>
      <span className="relative line-clamp-4 font-display text-lg font-bold leading-tight text-foreground/90">
        {movie.title}
      </span>
      <span className="relative text-[11px] text-muted-foreground">{movie.year || ""}</span>
    </div>
  );
}

/** Full-bleed backdrop; falls back to the generated gradient. */
export function Backdrop({ movie, className }: { movie: Movie; className?: string }) {
  const [error, setError] = useState(false);

  if (movie.backdropUrl && !error) {
    return (
      <img
        src={movie.backdropUrl}
        alt=""
        aria-hidden
        decoding="async"
        onError={() => setError(true)}
        className={cn("size-full object-cover", className)}
      />
    );
  }

  const hue = hueOf(movie);
  return (
    <div
      aria-hidden
      className={cn("size-full", className)}
      style={{
        background: `radial-gradient(120% 120% at 20% 20%, hsl(${hue} 40% 22%) 0%, hsl(${(hue + 30) % 360} 30% 10%) 45%, hsl(0 0% 5%) 100%)`,
      }}
    />
  );
}
