import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getMovie } from "@/api/movies";
import { getSimilarMovies } from "@/api/recommendations";
import { LikeButton, MyListButton, PlayButton } from "@/components/MovieActions";
import { MovieRow } from "@/components/MovieRow";
import { Backdrop, Poster } from "@/components/Poster";
import { RatingBadge } from "@/components/RatingBadge";
import { HeroSkeleton } from "@/components/LoadingSkeleton";
import { EmptyState } from "@/components/EmptyState";
import { ErrorState } from "@/components/ErrorState";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/movies/$movieId")({
  head: () => ({
    meta: [
      { title: "Movie details — Filmory" },
      {
        name: "description",
        content: "Full details, cast-free synopsis, rating and similar-title recommendations for this film.",
      },
      { property: "og:title", content: "Movie details — Filmory" },
      {
        property: "og:description",
        content: "Watch, like or save this title and see what Filmory recommends next.",
      },
    ],
  }),
  component: MovieDetailsPage,
});

function MovieDetailsPage() {
  const { movieId } = Route.useParams();
  const id = Number(movieId);

  const { data: movie, isLoading, isError, refetch } = useQuery({
    queryKey: ["movie", id],
    queryFn: () => getMovie(id),
  });
  const similar = useQuery({ queryKey: ["similar", id], queryFn: () => getSimilarMovies(id) });

  if (isLoading) return <HeroSkeleton />;

  if (isError) {
    return (
      <div className="mx-auto max-w-3xl px-4 pt-32">
        <ErrorState
          title="We couldn't load this movie"
          description="The catalog service didn't respond. Try again in a moment."
          onRetry={() => void refetch()}
        />
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="mx-auto max-w-3xl px-4 pt-32">
        <EmptyState
          title="We couldn't find that movie"
          description="It may have been removed from the catalog."
          action={
            <Button asChild className="rounded-none bg-primary">
              <Link to="/movies">Browse all movies</Link>
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="pb-12">
      <section className="relative w-full overflow-hidden">
        <div className="absolute inset-0 opacity-90">
          <Backdrop movie={movie} className="scale-110 blur-3xl saturate-200" />
        </div>
        <div className="hero-scrim absolute inset-0" />

        <div className="relative z-10 mx-auto flex max-w-[1600px] flex-col gap-8 px-4 pb-14 pt-28 md:flex-row md:px-10 md:pt-40">
          <div className="w-40 shrink-0 shadow-2xl shadow-background md:w-60">
            <Poster movie={movie} eager />
          </div>

          <div className="max-w-2xl">
            <h1 className="text-4xl font-extrabold md:text-6xl">{movie.title}</h1>

            <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-muted-foreground">
              <RatingBadge rating={movie.rating} />
              {movie.year ? <span>{movie.year}</span> : null}
              {movie.genres.length ? (
                <>
                  <span aria-hidden>•</span>
                  <span>{movie.genres.join(" • ")}</span>
                </>
              ) : null}
              {movie.runtime ? (
                <>
                  <span aria-hidden>•</span>
                  <span>
                    {Math.floor(movie.runtime / 60)}h {movie.runtime % 60}m
                  </span>
                </>
              ) : null}
            </div>

            <p className="mt-5 text-base leading-relaxed text-muted-foreground">
              {movie.description || "No synopsis is available for this title yet."}
            </p>

            <div className="mt-7 flex flex-wrap gap-3">
              <PlayButton movie={movie} />
              <LikeButton movie={movie} />
              <MyListButton movie={movie} />
            </div>

            <p className="mt-6 text-xs text-muted-foreground">
              MovieLens id <span className="font-mono text-gold">{movie.movieId}</span> — sent with every
              recorded interaction.
            </p>
          </div>
        </div>
      </section>

      <MovieRow
        title="You May Also Like"
        subtitle="Movie-to-movie recommendations"
        movies={similar.data ?? []}
        loading={similar.isLoading}
        error={similar.isError}
        onRetry={() => void similar.refetch()}
      />
    </div>
  );
}
