import { useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { toast } from "sonner";
import { getGenres, getMovies } from "@/api/movies";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ErrorState";
import { GridSkeleton } from "@/components/LoadingSkeleton";
import { useAuth } from "@/context/AuthContext";
import { cn } from "@/lib/utils";
import type { Movie } from "@/types/movie";

const STEPS = ["Welcome", "Genres", "Favourites", "Summary"] as const;

function toggle<T>(list: T[], value: T) {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

export function OnboardingWizard() {
  const { savePreferences } = useAuth();
  const navigate = useNavigate();
  const moviesQuery = useQuery({ queryKey: ["movies"], queryFn: getMovies });
  const genresQuery = useQuery({ queryKey: ["genres"], queryFn: getGenres });

  const [step, setStep] = useState(0);
  const [genres, setGenres] = useState<string[]>([]);
  const [movieIds, setMovieIds] = useState<number[]>([]);
  const [saving, setSaving] = useState(false);

  const movies = moviesQuery.data ?? [];
  const picked = useMemo(
    () => movies.filter((m) => movieIds.includes(m.movieId)),
    [movies, movieIds],
  );

  const canContinue = step === 0 || (step === 1 ? genres.length > 0 : true);

  const finish = async () => {
    setSaving(true);
    await savePreferences(genres, movieIds);
    toast.success("Preferences saved", { description: "Your home feed is ready." });
    void navigate({ to: "/" });
  };

  return (
    <div className="mx-auto max-w-4xl px-4 pb-24 pt-28 md:px-10">
      <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1 text-xs font-semibold text-gold">
        Step {step + 1} of {STEPS.length} — {STEPS[step]}
      </span>

      <div className="mt-4 flex gap-2" aria-hidden>
        {STEPS.map((label, i) => (
          <div
            key={label}
            className={cn(
              "h-1 flex-1 rounded-full transition-colors",
              i <= step ? "bg-primary" : "bg-surface-raised",
            )}
          />
        ))}
      </div>

      <div className="mt-8">
        {step === 0 ? <WelcomeStep /> : null}

        {step === 1 ? (
          <section>
            <h1 className="text-3xl font-extrabold md:text-4xl">Pick your genres</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Choose at least three so we can shape your first recommendations.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              {(genresQuery.data ?? []).map((genre) => {
                const active = genres.includes(genre);
                return (
                  <button
                    key={genre}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setGenres((g) => toggle(g, genre))}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-primary",
                      active
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-border bg-surface text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {active ? <Check className="size-3.5" aria-hidden /> : null}
                    {genre}
                  </button>
                );
              })}
            </div>
            <p className="mt-4 text-xs text-muted-foreground" aria-live="polite">
              {genres.length} genre{genres.length === 1 ? "" : "s"} selected
            </p>
          </section>
        ) : null}

        {step === 2 ? (
          <section>
            <h1 className="text-3xl font-extrabold md:text-4xl">Choose a few favourites</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Tap the posters you love — these seed the cold-start recommendations.
            </p>
            {moviesQuery.isError ? (
              <ErrorState
                className="mt-6"
                title="Couldn't load the catalog"
                onRetry={() => void moviesQuery.refetch()}
              />
            ) : moviesQuery.isLoading ? (
              <div className="mt-6">
                <GridSkeleton count={12} />
              </div>
            ) : (
              <div className="mt-6 grid max-h-[26rem] grid-cols-3 gap-3 overflow-y-auto pr-1 sm:grid-cols-5 lg:grid-cols-6">
                {movies.map((movie) => (
                  <PosterPick
                    key={movie.movieId}
                    movie={movie}
                    active={movieIds.includes(movie.movieId)}
                    onToggle={() => setMovieIds((ids) => toggle(ids, movie.movieId))}
                  />
                ))}
              </div>
            )}
            <p className="mt-4 text-xs text-muted-foreground" aria-live="polite">
              {movieIds.length} film{movieIds.length === 1 ? "" : "s"} selected
            </p>
          </section>
        ) : null}

        {step === 3 ? (
          <section>
            <h1 className="text-3xl font-extrabold md:text-4xl">Here's your taste profile</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              This is exactly what gets sent to the cold-start recommendation endpoint later.
            </p>

            <h2 className="mt-8 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Genres
            </h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {genres.length ? (
                genres.map((g) => (
                  <span
                    key={g}
                    className="rounded-md border border-primary/50 bg-primary/10 px-3 py-1 text-sm text-foreground"
                  >
                    {g}
                  </span>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">No genres picked yet.</span>
              )}
            </div>

            <h2 className="mt-8 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Favourite films
            </h2>
            <div className="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-6">
              {picked.length ? (
                picked.map((m) => (
                  <figure key={m.movieId}>
                    <img
                      src={m.posterUrl}
                      alt={`${m.title} poster`}
                      loading="lazy"
                      className="aspect-[2/3] w-full rounded-md object-cover"
                    />
                    <figcaption className="mt-1 truncate text-xs text-muted-foreground">
                      {m.title}
                    </figcaption>
                  </figure>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">No films picked yet.</span>
              )}
            </div>
          </section>
        ) : null}
      </div>

      <div className="mt-10 flex flex-wrap items-center gap-3">
        {step > 0 ? (
          <Button
            variant="secondary"
            onClick={() => setStep((s) => s - 1)}
            className="rounded-md border border-border bg-surface-raised/80 px-6 py-6"
          >
            <ArrowLeft className="size-4" aria-hidden />
            Back
          </Button>
        ) : null}

        {step < STEPS.length - 1 ? (
          <Button
            disabled={!canContinue}
            onClick={() => setStep((s) => s + 1)}
            className="rounded-md bg-primary px-8 py-6 font-semibold hover:bg-primary-glow"
          >
            {step === 0 ? "Get started" : "Continue"}
            <ArrowRight className="size-4" aria-hidden />
          </Button>
        ) : (
          <Button
            disabled={saving}
            onClick={() => void finish()}
            className="rounded-md bg-primary px-8 py-6 font-semibold hover:bg-primary-glow"
          >
            {saving ? "Saving…" : "Looks good"}
          </Button>
        )}

        <Button
          variant="ghost"
          onClick={() => navigate({ to: "/" })}
          className="rounded-md px-6 py-6 text-muted-foreground"
        >
          Skip for now
        </Button>
      </div>
    </div>
  );
}

function WelcomeStep() {
  return (
    <section>
      <h1 className="text-4xl font-extrabold md:text-5xl">Tell us what you like</h1>
      <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
        Three quick steps and your home feed stops guessing. Pick the genres you gravitate to, star a
        few favourites, and review your taste profile before we save it.
      </p>
      <ul className="mt-8 space-y-3 text-sm text-muted-foreground">
        {[
          "Pick the genres you watch most",
          "Star a handful of favourite films",
          "Review and confirm your profile",
        ].map((item, i) => (
          <li key={item} className="flex items-center gap-3">
            <span className="flex size-7 items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
              {i + 1}
            </span>
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

function PosterPick({
  movie,
  active,
  onToggle,
}: {
  movie: Movie;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onToggle}
      className={cn(
        "group relative overflow-hidden rounded-md border-2 text-left outline-none transition focus-visible:ring-2 focus-visible:ring-primary",
        active ? "border-primary" : "border-transparent hover:border-border",
      )}
    >
      <img
        src={movie.posterUrl}
        alt={`${movie.title} poster`}
        loading="lazy"
        className={cn(
          "aspect-[2/3] w-full rounded-sm object-cover transition",
          active ? "brightness-110" : "brightness-75 group-hover:brightness-100",
        )}
      />
      {active ? (
        <span className="absolute right-1.5 top-1.5 flex size-5 items-center justify-center rounded-sm bg-primary text-primary-foreground">
          <Check className="size-3" aria-hidden />
        </span>
      ) : null}
      <span className="block truncate px-1.5 py-1.5 text-xs">{movie.title}</span>
    </button>
  );
}
