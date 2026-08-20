import { createFileRoute, Link } from "@tanstack/react-router";
import { Play } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";
import { RequireAuth } from "@/components/RequireAuth";
import { Button } from "@/components/ui/button";
import { RatingBadge } from "@/components/RatingBadge";
import { useUserData } from "@/context/UserDataContext";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Watch history — Filmory" },
      { name: "description", content: "Everything you have played on Filmory, with dates and progress." },
      { property: "og:title", content: "Watch history — Filmory" },
      { property: "og:description", content: "Review and replay your Filmory viewing activity." },
    ],
  }),
  component: () => (
    <RequireAuth>
      <HistoryPage />
    </RequireAuth>
  ),
});

function HistoryPage() {
  const { history, play } = useUserData();

  return (
    <div className="mx-auto max-w-5xl px-4 pb-16 pt-24 md:px-10 md:pt-28">
      <h1 className="text-3xl font-extrabold md:text-4xl">Watch History</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Each play is recorded through the interactions API — future training signal for the model.
      </p>

      <div className="mt-8 space-y-3">
        {history.length === 0 ? (
          <EmptyState
            title="Nothing watched yet"
            description="Press Play on any movie and it will show up here."
            action={
              <Button asChild className="rounded-none bg-primary hover:bg-primary-glow">
                <Link to="/">Go to home</Link>
              </Button>
            }
          />
        ) : (
          history.map((entry) => (
            <article
              key={entry.movie.movieId}
              className="flex items-center gap-4 rounded-none border border-border bg-surface/70 p-3 transition-colors hover:bg-surface-raised/70"
            >
              <Link
                to="/movies/$movieId"
                params={{ movieId: String(entry.movie.movieId) }}
                className="shrink-0"
              >
                <img
                  src={entry.movie.posterUrl}
                  alt={`${entry.movie.title} poster`}
                  loading="lazy"
                  className="h-24 w-16 rounded-none object-cover"
                />
              </Link>

              <div className="min-w-0 flex-1">
                <Link
                  to="/movies/$movieId"
                  params={{ movieId: String(entry.movie.movieId) }}
                  className="truncate font-semibold hover:text-primary-glow"
                >
                  {entry.movie.title}
                </Link>
                <p className="mt-1 truncate text-xs text-muted-foreground">
                  {entry.movie.year} • {entry.movie.genres.join(" • ")}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Watched {new Date(entry.watchedAt).toLocaleString()}
                </p>
                <div className="mt-2 h-1 w-40 overflow-hidden rounded-none bg-muted">
                  <div className="h-full rounded-none bg-primary" style={{ width: `${entry.progress}%` }} />
                </div>
              </div>

              <RatingBadge rating={entry.movie.rating} className="hidden sm:inline-flex" />

              <Button
                onClick={() => void play(entry.movie)}
                className="rounded-none bg-primary hover:bg-primary-glow"
              >
                <Play className="size-4 fill-current" aria-hidden />
                Play again
              </Button>
            </article>
          ))
        )}
      </div>
    </div>
  );
}
