import { useEffect, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Poster } from "@/components/Poster";
import { useAuth } from "@/context/AuthContext";
import { startRecommendationStudy, saveStudyAnswer } from "@/api/recommendationStudy";
import type { RecommendationStudy, StudyChoice } from "@/api/recommendationStudy";

const choices: { value: StudyChoice; label: string }[] = [
  { value: "interesting", label: "Looks interesting" },
  { value: "not_for_me", label: "Not for me" },
  { value: "unsure", label: "Not sure" },
];

export function RecommendationReview() {
  const { user } = useAuth();
  const activeUserId = useRef(user?.userId);
  activeUserId.current = user?.userId;
  const [storedStudy, setStudy] = useState<RecommendationStudy | null>(null);
  const [studyOwner, setStudyOwner] = useState<string | undefined>();
  const study = studyOwner === user?.userId ? storedStudy : null;
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [finished, setFinished] = useState(false);
  useEffect(() => {
    setStudy(null);
    setStudyOwner(undefined);
    setError("");
    setFinished(false);
    setPending(false);
  }, [user?.userId]);
  if (!user) return null;

  const start = async () => {
    const owner = user.userId;
    setPending(true);
    setError("");
    try {
      const next = await startRecommendationStudy();
      if (activeUserId.current !== owner) return;
      setStudy(next);
      setStudyOwner(owner);
      setFinished(false);
    } catch (e) {
      if (activeUserId.current !== owner) return;
      setError(e instanceof Error ? e.message : "Could not start the review. Please retry.");
    } finally {
      if (activeUserId.current === owner) setPending(false);
    }
  };
  const answer = async (movieId: number, choice: StudyChoice) => {
    if (!study || pending) return;
    const owner = user.userId;
    setPending(true);
    setError("");
    try {
      const next = await saveStudyAnswer(study.id, movieId, choice);
      if (activeUserId.current !== owner) return;
      setStudy(next);
    } catch (e) {
      if (activeUserId.current !== owner) return;
      setError(e instanceof Error ? e.message : "Could not save your rating. Please retry.");
    } finally {
      if (activeUserId.current === owner) setPending(false);
    }
  };
  return (
    <section
      className="mx-4 my-6 rounded-xl border border-border bg-surface p-5 md:mx-10"
      aria-label="Recommendation feedback study"
    >
      <h2 className="text-lg font-bold">Help evaluate Filmory</h2>
      <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
        Optional project study: tell us which recommendations look interesting. Your answers and
        this movie list are saved against your account. Study exports omit names and emails. These
        ratings do not change your likes or watch history.
      </p>
      {error ? (
        <p role="alert" className="mt-3 text-sm text-destructive">
          {error}
        </p>
      ) : null}
      {finished ? (
        <p role="status" className="mt-4 text-sm text-gold">
          Thank you. Your ratings have been saved for the project study.
        </p>
      ) : !study ? (
        <Button className="mt-4" disabled={pending} onClick={() => void start()}>
          {pending ? "Loading recommendations…" : "Take part and review recommendations"}
        </Button>
      ) : (
        <div className="mt-4">
          <p role="status" className="mb-4 text-sm text-muted-foreground">
            {Object.keys(study.answers).length} of {study.movies.length} rated. Each choice is
            saved; use Not sure if you cannot judge a movie.
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            {study.movies.map((movie) => (
              <article
                key={movie.movieId}
                className="flex gap-3 rounded-lg border border-border p-3"
              >
                <Link
                  to="/movies/$movieId"
                  params={{ movieId: String(movie.movieId) }}
                  className="w-16 shrink-0"
                >
                  <Poster movie={movie} className="rounded-md" />
                </Link>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold">{movie.title}</h3>
                  <p className="mt-1 text-xs text-muted-foreground">{movie.genres.join(" • ")}</p>
                  <div
                    className="mt-3 flex flex-wrap gap-2"
                    role="group"
                    aria-label={`Rate ${movie.title}`}
                  >
                    {choices.map((choice) => (
                      <Button
                        key={choice.value}
                        size="sm"
                        variant={
                          study.answers[movie.movieId] === choice.value ? "default" : "secondary"
                        }
                        aria-pressed={study.answers[movie.movieId] === choice.value}
                        disabled={pending}
                        onClick={() => void answer(movie.movieId, choice.value)}
                      >
                        {choice.label}
                      </Button>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>
          <Button
            className="mt-4"
            disabled={pending || !Object.keys(study.answers).length}
            onClick={() => setFinished(true)}
          >
            Finish review
          </Button>
        </div>
      )}
    </section>
  );
}
