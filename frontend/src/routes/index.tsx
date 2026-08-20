import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getMovies } from "@/api/movies";
import { getByGenre, getPopular, getRecommendations, getTrending } from "@/api/recommendations";
import { HeroBanner } from "@/components/HeroBanner";
import { HeroSkeleton } from "@/components/LoadingSkeleton";
import { MovieRow } from "@/components/MovieRow";
import { useAuth } from "@/context/AuthContext";
import { useUserData } from "@/context/UserDataContext";
import type { ScoredMovie } from "@/types/movie";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Filmory — Personalised movie discovery" },
      {
        name: "description",
        content:
          "Browse recommended, trending and popular films in a cinematic interface built for an AI recommendation engine.",
      },
      { property: "og:title", content: "Filmory — Personalised movie discovery" },
      {
        property: "og:description",
        content: "Recommended For You, Trending Now, Popular Movies and more — powered by Filmory.",
      },
    ],
  }),
  component: Home,
});

function Home() {
  const { user } = useAuth();
  const { history } = useUserData();
  const userId = user?.userId ?? "guest";

  const recommendations = useQuery({
    queryKey: ["recommendations", userId],
    queryFn: () => getRecommendations(userId),
  });
  const trending = useQuery({ queryKey: ["trending"], queryFn: getTrending });
  const popular = useQuery({ queryKey: ["popular"], queryFn: getPopular });
  const catalog = useQuery({ queryKey: ["movies"], queryFn: getMovies });

  const heroMovies: ScoredMovie[] = (recommendations.data ?? catalog.data ?? []).slice(0, 5);
  const recentlyWatched = history.map((entry) => entry.movie);

  return (
    <div className="pb-10">
      {heroMovies.length ? <HeroBanner movies={heroMovies} /> : <HeroSkeleton />}

      <div className="relative z-10 -mt-4 md:-mt-6">
        <MovieRow
          title="Recommended For You"
          subtitle="Matched to your taste profile"
          movies={recommendations.data ?? []}
          loading={recommendations.isLoading}
          error={recommendations.isError}
          onRetry={() => void recommendations.refetch()}
        />

        <MovieRow
          title="Top 10 This Week"
          movies={(trending.data ?? []).slice(0, 10)}
          loading={trending.isLoading}
          error={trending.isError}
          onRetry={() => void trending.refetch()}
          ranked
        />

        <MovieRow
          title="Popular Movies"
          movies={popular.data ?? []}
          loading={popular.isLoading}
          error={popular.isError}
          onRetry={() => void popular.refetch()}
        />

        {GENRE_RAILS.map((g) => (
          <GenreRow key={g} genre={g} />
        ))}

        <MovieRow
          title="Recently Watched"
          movies={recentlyWatched}
          progressFor={(movie) =>
            history.find((h) => h.movie.movieId === movie.movieId)?.progress
          }
        />
      </div>
    </div>
  );
}


const GENRE_RAILS = ["Sci-Fi", "Comedy", "Thriller", "Animation", "Documentary"];

function GenreRow({ genre }: { genre: string }) {
  const query = useQuery({
    queryKey: ["genre", genre],
    queryFn: () => getByGenre(genre),
  });

  return (
    <MovieRow
      title={genre}
      subtitle={`Most-rated ${genre.toLowerCase()} titles in the catalog`}
      movies={query.data ?? []}
      loading={query.isLoading}
      error={query.isError}
      onRetry={() => void query.refetch()}
    />
  );
}
