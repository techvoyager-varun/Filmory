import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { searchMovies } from "@/api/movies";
import { getTrending } from "@/api/recommendations";
import { SearchBar } from "@/components/SearchBar";
import { MovieGrid } from "@/components/MovieGrid";
import { MovieRow } from "@/components/MovieRow";
import { GridSkeleton } from "@/components/LoadingSkeleton";
import { EmptyState } from "@/components/EmptyState";
import { ErrorState } from "@/components/ErrorState";

export const Route = createFileRoute("/search")({
  head: () => ({
    meta: [
      { title: "Search movies — Filmory" },
      {
        name: "description",
        content: "Search the Filmory catalog by title, genre or year and open any film instantly.",
      },
      { property: "og:title", content: "Search movies — Filmory" },
      { property: "og:description", content: "Find any title in the Filmory catalog as you type." },
    ],
  }),
  component: SearchPage,
});

const SUGGESTIONS = ["Interstellar", "Sci-Fi", "Animation", "1994", "Thriller"];

function SearchPage() {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");

  useEffect(() => {
    const id = setTimeout(() => setDebounced(query), 280);
    return () => clearTimeout(id);
  }, [query]);

  const { data, isFetching, isError, refetch } = useQuery({
    queryKey: ["search", debounced],
    queryFn: () => searchMovies(debounced),
    enabled: debounced.trim().length > 0,
  });

  const trending = useQuery({ queryKey: ["trending"], queryFn: getTrending });
  const hasQuery = debounced.trim().length > 0;

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-16 pt-24 md:px-10 md:pt-28">
      <h1 className="text-3xl font-extrabold md:text-4xl">Search</h1>

      <div className="mt-6 max-w-2xl">
        <SearchBar value={query} onChange={setQuery} autoFocus />
        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setQuery(s)}
              className="rounded-md border border-border bg-surface px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-10">
        {!hasQuery ? (
          <MovieRow
            title="Trending Now"
            movies={trending.data ?? []}
            loading={trending.isLoading}
            error={trending.isError}
            onRetry={() => void trending.refetch()}
          />
        ) : isError ? (
          <ErrorState
            title="Search is unavailable"
            description="We couldn't reach the catalog. Try your search again."
            onRetry={() => void refetch()}
          />
        ) : isFetching ? (
          <GridSkeleton count={6} />
        ) : data && data.length > 0 ? (
          <>
            <p className="mb-6 text-sm text-muted-foreground" aria-live="polite">
              {data.length} result{data.length === 1 ? "" : "s"} for "{debounced}"
            </p>
            <MovieGrid movies={data} />
          </>
        ) : (
          <EmptyState
            title={`No matches for "${debounced}"`}
            description="Try another title, a genre like Sci-Fi, or a release year."
          />
        )}
      </div>
    </div>
  );
}
