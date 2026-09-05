import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Brain,
  Gauge,
  Layers,
  ListOrdered,
  MoveRight,
  Radar,
  Sparkles,
  TrendingUp,
  Wand2,
} from "lucide-react";
import { getModelMetrics } from "@/api/recommendations";
import type { ModelMetrics } from "@/types/movie";

export const Route = createFileRoute("/model")({
  head: () => ({
    meta: [
      { title: "How Filmory Recommends — the 4-Stage Engine" },
      {
        name: "description",
        content:
          "Filmory's 4-stage recommendation pipeline: NCF Hybrid candidates, Sequential Transformer scoring, genre affinity, and the DAMR drift-aware re-ranker — with measured accuracy.",
      },
    ],
  }),
  component: ModelPage,
});

export default function ModelPage() {
  const metrics = useQuery({
    queryKey: ["model-metrics"],
    queryFn: getModelMetrics,
    staleTime: 5 * 60 * 1000,
  });

  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-20 pt-24 md:px-8 md:pt-28">
      {/* ------------------------------------------------ header */}
      <header className="border-l-2 border-gold/70 pl-4">
        <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-gold">
          <Sparkles className="size-3.5" /> AI transparency
        </p>
        <h1 className="mt-2 font-display text-3xl font-extrabold tracking-tight md:text-4xl">
          How Filmory Recommends
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground md:text-base">
          Every recommendation you see is produced by a 4-stage pipeline that fuses three trained
          neural experts with a drift-aware re-ranker. This page shows exactly how it works — and
          how accurate it is, measured with the standard leave-one-out protocol used by the NCF and
          SASRec papers.
        </p>
      </header>

      {/* ------------------------------------------------ pipeline */}
      <PipelineSection />

      {/* ------------------------------------------------ DAMR */}
      <DamrSection />

      {/* ------------------------------------------------ metrics */}
      <MetricsSection
        metrics={metrics.data}
        loading={metrics.isLoading}
        error={metrics.isError}
        onRetry={() => void metrics.refetch()}
      />

      {/* ------------------------------------------------ glossary */}
      <GlossarySection />

      {/* ------------------------------------------------ score fields */}
      <ScoreFieldsSection />
    </div>
  );
}

/* ------------------------------------------------------------------ pipeline */
const STAGES = [
  {
    id: 1,
    name: "NCF Hybrid",
    role: "Candidate generation",
    body: "A hybrid Neural Collaborative Filtering network scores all 22,836 unseen movies for you in one forward pass and keeps the top 100. Learned 8-dim user & item embeddings + genre projections capture who you are.",
    detail: "Stage 1 expert → ncfScore",
  },
  {
    id: 2,
    name: "Sequential Transformer",
    role: "Sequence scoring",
    body: "A 2-layer, 4-head transformer (SASRec-style) reads your last 20 watches in order, producing a 'what does this user want next' vector. Each candidate is scored by cosine similarity against it.",
    detail: "Stage 2 expert → transformerScore",
  },
  {
    id: 3,
    name: "Genre Affinity",
    role: "Real-time signal",
    body: "Your live genre preference vector (accumulated from plays, likes and onboarding in PostgreSQL) is dotted with every candidate's genre vector — the freshest signal in the pipeline.",
    detail: "Stage 3 expert → genreScore",
  },
  {
    id: 4,
    name: "DAMR",
    role: "Drift-Aware Momentum Re-Ranker",
    body: "Filmory's own Stage 4. It estimates how much your taste is moving right now, re-weights the three experts per request, adds a momentum bonus toward the direction of change, an agreement-confidence factor, a quality prior, and MMR diversity — then picks the final 10.",
    detail: "Stage 4 → score",
    highlight: true,
  },
];

function PipelineSection() {
  return (
    <section className="mt-12">
      <SectionTitle icon={<Layers className="size-4" />} title="The 4-stage pipeline" />
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        {STAGES.map((s, i) => (
          <div
            key={s.id}
            className={`relative rounded-xl border p-5 ${
              s.highlight
                ? "border-gold/50 bg-gold/5 shadow-[0_0_40px_-16px_var(--gold)]"
                : "border-border bg-surface/60"
            }`}
          >
            <div className="flex items-center gap-3">
              <span
                className={`flex size-9 items-center justify-center rounded-lg font-display text-sm font-extrabold ${
                  s.highlight
                    ? "bg-gold text-gold-foreground"
                    : "bg-primary text-primary-foreground"
                }`}
              >
                {s.id}
              </span>
              <div>
                <h3 className="font-display text-base font-bold leading-tight">{s.name}</h3>
                <p className="text-xs text-muted-foreground">{s.role}</p>
              </div>
              {i < STAGES.length - 1 && (
                <MoveRight className="ml-auto hidden size-5 text-muted-foreground/50 md:block" />
              )}
            </div>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{s.body}</p>
            <p
              className={`mt-3 font-mono text-[11px] ${s.highlight ? "text-gold" : "text-gold/70"}`}
            >
              {s.detail}
            </p>
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-xl border border-border bg-surface/40 p-4 font-mono text-xs leading-relaxed text-muted-foreground">
        candidates&nbsp;&nbsp;= NCF_top100(cat · 22,836 items)
        <br />
        final(i)&nbsp;&nbsp;&nbsp;&nbsp;= w<sub>ncf</sub>·s<sub>ncf</sub>(i) + w<sub>tr</sub>·s
        <sub>tr</sub>(i) + w<sub>gen</sub>·s<sub>gen</sub>(i) + η·δ·momentum(i)
        <br />
        w&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
        softmax(gate&nbsp;|&nbsp;drift&nbsp;δ, focus&nbsp;φ, maturity&nbsp;μ,
        freshness&nbsp;f)&nbsp;&nbsp;←&nbsp;replaces the fixed 0.55 / 0.25 / 0.20
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ DAMR */
function DamrSection() {
  const states = [
    {
      symbol: "δ",
      name: "Drift",
      formula: "1 − cos(G_short, G_long)",
      body: "How far your current taste has moved from your lifetime identity. Time-decayed genre profile (τ = 7 days) vs. the uniform profile over all your watches.",
    },
    {
      symbol: "φ",
      name: "Focus",
      formula: "1 − H(G_short) / log 20",
      body: "Is your current interest concentrated on a few genres (binging) or scattered? Entropy of the decayed profile.",
    },
    {
      symbol: "μ",
      name: "Maturity",
      formula: "log(1+n) / log(1+200)",
      body: "How much history you have — a proxy for how much the long-term NCF embedding can be trusted.",
    },
    {
      symbol: "f",
      name: "Freshness",
      formula: "exp(−hours_since_last / 48)",
      body: "Are you in an active session right now? Fresh activity up-weights the sequential expert.",
    },
  ];
  return (
    <section className="mt-14">
      <SectionTitle icon={<Wand2 className="size-4" />} title="Stage 4 up close: DAMR" />
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Fixed fusion weights are wrong for many users: a horror-bingeing comedy fan should
        temporarily trust the sequence expert, a 500-watch veteran should trust the NCF embedding, a
        brand-new account should lean on genres. DAMR measures four interpretable state scalars from
        your history and computes the expert weights <em>per request</em> — the old fixed ensemble
        is the provable special case at the calibration anchor. It also adds the one thing NCF and
        the Transformer cannot express: the <em>direction</em> your taste is moving.
      </p>
      <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {states.map((s) => (
          <div key={s.symbol} className="rounded-xl border border-border bg-surface/60 p-4">
            <div className="flex items-baseline gap-2">
              <span className="font-display text-2xl font-extrabold text-gold">{s.symbol}</span>
              <h3 className="font-display text-sm font-bold">{s.name}</h3>
            </div>
            <p className="mt-2 font-mono text-[11px] text-muted-foreground">{s.formula}</p>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{s.body}</p>
          </div>
        ))}
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <DamrStep
          icon={<TrendingUp className="size-4" />}
          title="Taste-momentum term"
          body="M = G_short − G_long is the direction your taste is moving. Candidates aligned with M get a bonus scaled by drift: η·δ·cos(genre, M). Only this term extrapolates where you're going, not where you've been."
        />
        <DamrStep
          icon={<Brain className="size-4" />}
          title="Expert-agreement confidence"
          body="Movies all three experts agree on get a small boost; movies only one expert likes get damped. Cheap, interpretable, and it visibly stabilises the list."
        />
        <DamrStep
          icon={<ListOrdered className="size-4" />}
          title="Quality prior + MMR diversity"
          body="An IMDb-style Bayesian weighted rating keeps obscure 2-star titles out, and Maximal Marginal Relevance over NCF-embedding + genre similarity stops the top-10 being 10 near-identical movies."
        />
      </div>
    </section>
  );
}

function DamrStep({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface/60 p-4">
      <h3 className="flex items-center gap-2 font-display text-sm font-bold text-gold">
        {icon}
        {title}
      </h3>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ metrics */
const SYSTEM_DESCRIPTIONS: Record<string, string> = {
  Popularity: "Non-personalised baseline — most interacted titles.",
  "NCF Baseline": "Pure collaborative filtering (Stage 1's backbone).",
  "NCF Hybrid": "NCF + genre projections — the Stage 1 expert.",
  Transformer: "Sequential expert (Stage 2).",
  Genre: "Real-time genre-affinity expert (Stage 3).",
  "Static Ensemble": "Fixed 0.55 / 0.25 / 0.20 fusion (the legacy system).",
  "Static + MMR": "Fixed fusion + quality prior + MMR diversity.",
  "DAMR (ours)":
    "Stage 4: drift-aware adaptive fusion + momentum + agreement + quality + diversity.",
};

function MetricsSection({
  metrics,
  loading,
  error,
  onRetry,
}: {
  metrics: ModelMetrics | undefined;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
}) {
  const k = metrics?.k ?? 10;
  const systems = metrics ? Object.keys(metrics.results) : [];
  const hrKey = `HR@${k}`;
  const maxHr = metrics
    ? Math.max(...systems.map((s) => metrics.results[s]?.[hrKey] ?? 0), 0.0001)
    : 1;

  return (
    <section className="mt-14">
      <SectionTitle icon={<Gauge className="size-4" />} title="Measured accuracy" />
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Implicit-feedback recommenders aren't measured with classification accuracy (99.9% of
        user–movie pairs are negatives, so predicting "no" everywhere is trivially "accurate"). The
        standard is hit-rate based ranking measurement — here is Filmory's, evaluated offline.
      </p>

      {loading ? (
        <div className="mt-5 h-40 animate-pulse rounded-xl bg-surface/60" />
      ) : error || !metrics ? (
        <div className="mt-5 rounded-xl border border-border bg-surface/60 p-6 text-sm text-muted-foreground">
          <p className="font-medium text-foreground">Metrics not generated yet</p>
          <p className="mt-1">
            Run{" "}
            <code className="rounded bg-surface-raised px-1.5 py-0.5 text-gold">
              cd backend &amp;&amp; python scripts/evaluate.py
            </code>{" "}
            to produce{" "}
            <code className="rounded bg-surface-raised px-1.5 py-0.5 text-gold">
              ml/metrics.json
            </code>
            , then reload.
          </p>
          <button
            onClick={onRetry}
            className="mt-3 rounded-md border border-border px-3 py-1.5 text-xs hover:bg-surface-raised"
          >
            Retry
          </button>
        </div>
      ) : (
        <>
          <p className="mt-4 rounded-lg border border-border bg-surface/40 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
            protocol: {metrics.protocol} · users evaluated: {metrics.num_users_evaluated} ·
            catalogue: {metrics.dataset.num_items.toLocaleString()} items ·{" "}
            {new Date(metrics.generated_at).toISOString().slice(0, 10)}
          </p>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead>
                <tr className="border-b border-border bg-surface-raised/60 text-xs uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-3 font-semibold">System</th>
                  <th className="px-4 py-3 font-semibold">HR@{k} ↑</th>
                  <th className="px-4 py-3 font-semibold">NDCG@{k} ↑</th>
                  <th className="px-4 py-3 font-semibold">MRR ↑</th>
                  <th className="px-4 py-3 font-semibold">AUC ↑</th>
                  <th className="px-4 py-3 font-semibold">ILD@{k} ↑</th>
                  <th className="px-4 py-3 font-semibold">Coverage ↑</th>
                </tr>
              </thead>
              <tbody>
                {systems.map((system) => {
                  const r = metrics.results[system] ?? {};
                  const isOurs = system.startsWith("DAMR");
                  const hr = (r[hrKey] as number) ?? 0;
                  return (
                    <tr
                      key={system}
                      className={`border-b border-border/60 last:border-0 ${
                        isOurs ? "bg-gold/5" : ""
                      }`}
                    >
                      <td className="px-4 py-3">
                        <div className={`font-medium ${isOurs ? "text-gold" : ""}`}>{system}</div>
                        <div className="text-[11px] text-muted-foreground">
                          {SYSTEM_DESCRIPTIONS[system] ?? ""}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="w-12 font-mono text-xs">{hr.toFixed(4)}</span>
                          <span className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-surface-raised sm:block">
                            <span
                              className={`block h-full rounded-full ${isOurs ? "bg-gold" : "bg-primary/70"}`}
                              style={{ width: `${(hr / maxHr) * 100}%` }}
                            />
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">{fmt(r[`NDCG@${k}`])}</td>
                      <td className="px-4 py-3 font-mono text-xs">{fmt(r["MRR"])}</td>
                      <td className="px-4 py-3 font-mono text-xs">{fmt(r["AUC"])}</td>
                      <td className="px-4 py-3 font-mono text-xs">{fmt(r[`ILD@${k}`])}</td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {r[`Coverage@${k}`] != null
                          ? `${((r[`Coverage@${k}`] as number) * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            Reading the table: popularity is a strong baseline under uniformly sampled negatives (it
            is rarely punished by unseen titles), which is why the NCF paper reports it too. Among
            the personalised systems, the full 4-stage DAMR pipeline gives the best hit rate, and
            the diversity columns show the MMR/quality components trading almost no relevance for
            list variety.
          </p>
        </>
      )}
    </section>
  );
}

function fmt(v: number | null | undefined): string {
  return typeof v === "number" ? v.toFixed(4) : "—";
}

/* ------------------------------------------------------------------ glossary */
const GLOSSARY = [
  {
    term: "HR@K — Hit Rate",
    body: "Of all evaluated users, in how many cases did the one movie they actually watched next appear inside the top-K? The headline 'did we get it right' number.",
  },
  {
    term: "NDCG@K",
    body: "Like HR, but a hit at rank 1 scores much higher than a hit at rank 10 (logarithmic discount). Measures ranking quality, not just inclusion.",
  },
  {
    term: "MRR — Mean Reciprocal Rank",
    body: "Average of 1 / rank of the held-out movie. Sensitive to how high the correct answer lands.",
  },
  {
    term: "AUC",
    body: "Probability the model scores the true positive above a random negative. Threshold-free measure of discrimination.",
  },
  {
    term: "ILD@K — Intra-List Diversity",
    body: "Average pairwise dissimilarity (NCF embedding + genre cosine) inside the top-K. Higher = a list that spans more of your taste.",
  },
  {
    term: "Catalogue Coverage",
    body: "Share of all movies that appear in someone's top-K. Higher = the engine surfaces the long tail, not just blockbusters.",
  },
];

function GlossarySection() {
  return (
    <section className="mt-14">
      <SectionTitle icon={<Radar className="size-4" />} title="Metric glossary" />
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        {GLOSSARY.map((g) => (
          <div key={g.term} className="rounded-xl border border-border bg-surface/60 p-4">
            <h3 className="font-display text-sm font-bold text-gold">{g.term}</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{g.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ score fields */
function ScoreFieldsSection() {
  return (
    <section className="mt-14">
      <SectionTitle
        icon={<Brain className="size-4" />}
        title="What each score on a movie card means"
      />
      <div className="mt-5 overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead>
            <tr className="border-b border-border bg-surface-raised/60 text-xs uppercase tracking-wider text-muted-foreground">
              <th className="px-4 py-3 font-semibold">Field</th>
              <th className="px-4 py-3 font-semibold">Meaning</th>
            </tr>
          </thead>
          <tbody className="text-xs">
            {[
              ["score", "Final relevance after Stage 4 (DAMR) re-ranking — the match %."],
              ["ncfScore", "Stage 1: long-term taste identity from the NCF Hybrid expert."],
              ["transformerScore", "Stage 2: fit with your last-20-watch sequence."],
              ["genreScore", "Stage 3: alignment with your live genre-preference vector."],
              ["momentumScore", "Stage 4: alignment with the direction your taste is moving."],
              ["agreementScore", "Stage 4: how strongly all three experts agree on this movie."],
              ["qualityScore", "Stage 4: Bayesian-weighted community rating prior."],
              ["diversityPenalty", "Stage 4: similarity to the movies already picked above it."],
              [
                "expertWeights",
                "The per-request gate output — how much each expert was trusted right now.",
              ],
            ].map(([field, meaning]) => (
              <tr key={field} className="border-b border-border/60 last:border-0">
                <td className="px-4 py-2.5 font-mono text-gold">{field}</td>
                <td className="px-4 py-2.5 text-muted-foreground">{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-3 border-l-2 border-gold/70 pl-3">
      <span className="flex size-8 items-center justify-center rounded-lg bg-surface-raised text-gold">
        {icon}
      </span>
      <h2 className="font-display text-xl font-extrabold tracking-tight md:text-2xl">{title}</h2>
    </div>
  );
}
