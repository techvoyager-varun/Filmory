import { createFileRoute, Link } from "@tanstack/react-router";
import { EmptyState } from "@/components/EmptyState";
import { GridSkeleton } from "@/components/LoadingSkeleton";
import { MovieGrid } from "@/components/MovieGrid";
import { RequireAuth } from "@/components/RequireAuth";
import { Button } from "@/components/ui/button";
import { useUserData } from "@/context/UserDataContext";

export const Route = createFileRoute("/my-list")({
  head: () => ({
    meta: [
      { title: "My List — Filmory" },
      { name: "description", content: "Every film you saved to watch later on Filmory." },
      { property: "og:title", content: "My List — Filmory" },
      { property: "og:description", content: "Your saved Filmory watchlist in one place." },
    ],
  }),
  component: () => (
    <RequireAuth>
      <MyListPage />
    </RequireAuth>
  ),
});

function MyListPage() {
  const { myList, loading, removeFromList } = useUserData();

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-16 pt-24 md:px-10 md:pt-28">
      <h1 className="text-3xl font-extrabold md:text-4xl">My List</h1>
      <p className="mt-2 text-sm text-muted-foreground">Saved titles, ready when you are.</p>

      <div className="mt-8">
        {loading ? (
          <GridSkeleton count={6} />
        ) : myList.length ? (
          <MovieGrid movies={myList} onRemove={(movie) => void removeFromList(movie)} />
        ) : (
          <EmptyState
            title="Your list is empty"
            description="Add movies with the + button on any card to build your watchlist."
            action={
              <Button asChild className="rounded-md bg-primary hover:bg-primary-glow">
                <Link to="/movies">Browse movies</Link>
              </Button>
            }
          />
        )}
      </div>
    </div>
  );
}
