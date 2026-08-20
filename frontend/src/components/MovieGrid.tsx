import { MovieCard } from "@/components/MovieCard";
import type { ScoredMovie } from "@/types/movie";

export function MovieGrid({
  movies,
  onRemove,
  subtitleFor,
}: {
  movies: ScoredMovie[];
  onRemove?: (movie: ScoredMovie) => void;
  subtitleFor?: (movie: ScoredMovie) => string | undefined;
}) {
  return (
    <div className="grid grid-cols-2 justify-items-center gap-x-4 gap-y-6 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
      {movies.map((movie) => (
        <MovieCard
          key={movie.movieId}
          movie={movie}
          subtitle={subtitleFor?.(movie)}
          onRemove={onRemove ? () => onRemove(movie) : undefined}
          className="w-full max-w-[190px]"
        />
      ))}
    </div>
  );
}
