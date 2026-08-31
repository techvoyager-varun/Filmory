import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getRecommendations } from "@/api/recommendations";
import { MovieRow } from "@/components/MovieRow";
import { RequireAuth } from "@/components/RequireAuth";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";
import { useUserData } from "@/context/UserDataContext";

export const Route = createFileRoute("/profile")({
  head: () => ({
    meta: [
      { title: "Your profile — Filmory" },
      {
        name: "description",
        content: "Your Filmory account, favourite genres, saved list, history and recommendations.",
      },
      { property: "og:title", content: "Your profile — Filmory" },
      { property: "og:description", content: "Manage your Filmory taste profile and activity." },
    ],
  }),
  component: () => (
    <RequireAuth>
      <ProfilePage />
    </RequireAuth>
  ),
});

function ProfilePage() {
  const { user, logout } = useAuth();
  const { myList, history } = useUserData();
  const recommendations = useQuery({
    queryKey: ["recommendations", user?.userId ?? "guest"],
    queryFn: () => getRecommendations(user?.userId ?? "guest"),
  });

  if (!user) return null;

  return (
    <div className="pb-14 pt-24 md:pt-28">
      <section className="mx-auto max-w-[1600px] px-4 md:px-10">
        <div className="flex flex-col gap-6 rounded-xl border border-border bg-surface/70 p-6 shadow-xl md:flex-row md:items-center md:p-8">
          <span className="flex size-20 items-center justify-center rounded-full bg-primary text-3xl font-extrabold text-primary-foreground shadow-md">
            {user.name.charAt(0).toUpperCase()}
          </span>

          <div className="flex-1">
            <h1 className="text-2xl font-extrabold md:text-3xl">{user.name}</h1>
            <p className="text-sm text-muted-foreground">{user.email}</p>

            <div className="mt-4 flex flex-wrap gap-2">
              {user.favoriteGenres.length ? (
                user.favoriteGenres.map((g) => (
                  <span
                    key={g}
                    className="rounded-md border border-border bg-surface-raised px-3 py-1 text-xs text-gold"
                  >
                    {g}
                  </span>
                ))
              ) : (
                <Link to="/onboarding" className="text-xs text-gold hover:underline">
                  Add favourite genres
                </Link>
              )}
            </div>
          </div>

          <div className="flex gap-6 text-center">
            <Stat label="My List" value={myList.length} />
            <Stat label="Watched" value={history.length} />
          </div>

          <Button variant="secondary" className="rounded-md" onClick={logout}>
            Sign out
          </Button>
        </div>
      </section>

      <div className="mt-6">
        <MovieRow
          title="Recommended For You"
          subtitle="Picked from your ratings and watch history"
          movies={recommendations.data ?? []}
          loading={recommendations.isLoading}
          error={recommendations.isError}
          onRetry={() => void recommendations.refetch()}
        />
        <MovieRow title="My List" movies={myList} />
        <MovieRow
          title="Recently Watched"
          movies={history.map((h) => h.movie)}
          progressFor={(movie) => history.find((h) => h.movie.movieId === movie.movieId)?.progress}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <p className="font-display text-2xl font-extrabold">{value}</p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}
