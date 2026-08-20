import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { browseMovies, getGenres } from "@/api/movies";
import { GridSkeleton } from "@/components/LoadingSkeleton";
import { MovieGrid } from "@/components/MovieGrid";
import { ErrorState } from "@/components/ErrorState";
import { EmptyState } from "@/components/EmptyState";
import { SearchBar } from "@/components/SearchBar";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/movies/")({
  head: () => ({
    meta: [
      { title: "Browse all movies — Filmory" },
      {
        name: "description",
        content:
          "Explore the full 27,000-title Filmory catalog and filter films by genre, rating or release year.",
      },
      { property: "og:title", content: "Browse all movies — Filmory" },
      { property: "og:description", content: "Filter the Filmory catalog by genre, rating and year." },
    ],
  }),
  component: MoviesPage,
});

type Sort = "popular" | "rating" | "year" | "title";
const PAGE_SIZE = 48;

function MoviesPage() {
  const [genre, setGenre] = useState("All");
  const [sort, setSort] = useState<Sort>("popular");
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [limit, setLimit] = useState(PAGE_SIZE);

  useEffect(() => {
    const id = setTimeout(() => setDebounced(query), 260);
    return () => clearTimeout(id);
  }, [query]);

  useEffect(() => setLimit(PAGE_SIZE), [genre, sort, debounced]);

  const { data, isLoading, isFetching, isError, refetch } = useQuery({
    queryKey: ["browse", genre, sort, debounced, limit],
    queryFn: () => browseMovies({ genre, sort, query: debounced, limit }),
    placeholderData: keepPreviousData,
  });

  const genresQuery = useQuery({ queryKey: ["genres"], queryFn: getGenres });
  const genreOptions = ["All", ...(genresQuery.data ?? [])];

  const movies = data?.movies ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-16 pt-24 md:px-10 md:pt-28">
      <h1 className="text-3xl font-extrabold md:text-4xl">All Movies</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        {isLoading ? "Loading catalog..." : `${total.toLocaleString()} titles match your filters`}
      </p>

      <div className="mt-6 max-w-xl">
        <SearchBar value={query} onChange={setQuery} placeholder="Filter titles in the catalog" />
      </div>

      <div className="scrollbar-none mt-6 flex gap-2 overflow-x-auto pb-2">
        {genreOptions.map((g) => (
          <button
            key={g}
            type="button"
            onClick={() => setGenre(g)}
            aria-pressed={genre === g}
            className={cn(
              "shrink-0 rounded-none border px-3.5 py-1.5 text-sm transition-colors",
              genre === g
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-surface text-muted-foreground hover:text-foreground",
            )}
          >
            {g}
          </button>
        ))}
      </div>

      <div className="mt-4 flex gap-2 text-sm">
        <span className="self-center text-muted-foreground">Sort by</span>
        {(["popular", "rating", "year", "title"] as Sort[]).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSort(s)}
            aria-pressed={sort === s}
            className={cn(
              "rounded-none px-3 py-1 capitalize transition-colors",
              sort === s ? "bg-surface-raised text-gold" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="mt-8">
        {isError ? (
          <ErrorState
            title="Couldn't load the catalog"
            description="The movie service didn't respond. Try again in a moment."
            onRetry={() => void refetch()}
          />
        ) : isLoading ? (
          <GridSkeleton />
        ) : movies.length ? (
          <>
            <MovieGrid movies={movies} />
            {movies.length < total ? (
              <div className="mt-10 flex justify-center">
                <Button
                  variant="secondary"
                  className="rounded-none border border-border bg-surface-raised px-8"
                  disabled={isFetching}
                  onClick={() => setLimit((l) => l + PAGE_SIZE)}
                >
                  {isFetching ? "Loading..." : `Load more (${(total - movies.length).toLocaleString()} left)`}
                </Button>
              </div>
            ) : null}
          </>
        ) : (
          <EmptyState title="No movies match" description="Try a different genre or search term." />
        )}
      </div>
    </div>
  );
}
