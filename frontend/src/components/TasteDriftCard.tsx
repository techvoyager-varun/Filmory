import { useQuery } from "@tanstack/react-query";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Activity, RefreshCw } from "lucide-react";
import { getTasteState } from "@/api/recommendations";
import type { UserStateInfo } from "@/types/movie";

/**
 * Live DAMR state for the logged-in user: the four taste-state scalars, the
 * drift-adaptive expert weights currently in effect, and a radar chart of the
 * lifetime vs. time-decayed genre profiles (the two vectors whose comparison
 * produces the drift estimate and the momentum direction).
 */
export function TasteDriftCard() {
  const taste = useQuery({
    queryKey: ["taste-state"],
    queryFn: getTasteState,
    staleTime: 30 * 1000,
    retry: false,
  });

  if (taste.isError) return null; // anonymous sessions simply don't show the card

  const state: UserStateInfo | undefined = taste.data?.state;
  const vectors = taste.data?.vectors;
  const genres = taste.data?.genres ?? [];

  const radarData =
    vectors && genres.length === vectors.long.length
      ? genres.map((g, i) => ({
          genre: g,
          lifetime: round4(vectors.long[i] ?? 0),
          recent: round4(vectors.short[i] ?? 0),
        }))
      : [];

  return (
    <section className="mt-6 rounded-xl border border-border bg-surface/70 p-5 shadow-lg md:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/15 text-primary-glow">
            <Activity className="size-4.5" />
          </span>
          <div>
            <h2 className="font-display text-lg font-extrabold tracking-tight">
              Taste Drift — live DAMR state
            </h2>
            <p className="text-xs text-muted-foreground">
              Stage 4 estimates these four numbers on every recommendation request.
            </p>
          </div>
        </div>
        <button
          onClick={() => void taste.refetch()}
          className="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-surface-raised hover:text-foreground"
        >
          <RefreshCw className={`size-3.5 ${taste.isFetching ? "animate-spin" : ""}`} />
          Recompute
        </button>
      </div>

      {taste.isLoading || !state ? (
        <div className="mt-5 h-44 animate-pulse rounded-lg bg-surface-raised/50" />
      ) : (
        <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_1fr]">
          {/* state scalars + weights */}
          <div>
            <div className="grid grid-cols-2 gap-3">
              <StateGauge
                label="Drift δ"
                value={state.drift}
                hint="how far current taste moved from identity"
                accent
              />
              <StateGauge
                label="Focus φ"
                value={state.focus}
                hint="concentration of current interest"
              />
              <StateGauge
                label="Maturity μ"
                value={state.maturity}
                hint="history depth — NCF reliability"
              />
              <StateGauge
                label="Freshness f"
                value={state.freshness}
                hint="active-session signal"
              />
            </div>

            <div className="mt-4 rounded-lg border border-border bg-surface-raised/40 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Expert weights this request
              </p>
              <WeightBar label="NCF Hybrid" value={state.weights.ncf} className="bg-primary" />
              <WeightBar
                label="Transformer"
                value={state.weights.transformer}
                className="bg-gold"
              />
              <WeightBar label="Genre" value={state.weights.genre} className="bg-emerald-500" />
              <p className="mt-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {state.nInteractions} interactions analysed · fixed legacy weights were 55% / 25% /
                20% — DAMR adapts them to your current state.
              </p>
            </div>
          </div>

          {/* radar: lifetime vs decayed profile */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Genre profile — lifetime vs last ~2 weeks
            </p>
            <div className="mt-2 h-64 w-full">
              {radarData.length > 0 && (
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="72%">
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis
                      dataKey="genre"
                      tick={{ fill: "var(--muted-foreground)", fontSize: 9 }}
                    />
                    <PolarRadiusAxis tick={false} axisLine={false} />
                    <Radar
                      name="Lifetime"
                      dataKey="lifetime"
                      stroke="var(--muted-foreground)"
                      fill="var(--muted-foreground)"
                      fillOpacity={0.18}
                    />
                    <Radar
                      name="Recent (decayed)"
                      dataKey="recent"
                      stroke="var(--gold)"
                      fill="var(--gold)"
                      fillOpacity={0.32}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--surface-raised)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              )}
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              The gap between the two polygons <em>is</em> your drift δ, and their difference is the
              momentum vector that pushes recommendations toward what you're moving toward.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

function StateGauge({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: number;
  hint: string;
  accent?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="rounded-lg border border-border bg-surface-raised/40 p-3">
      <div className="flex items-baseline justify-between">
        <span className={`text-xs font-semibold ${accent ? "text-gold" : "text-foreground"}`}>
          {label}
        </span>
        <span className="font-mono text-sm font-bold">{value.toFixed(2)}</span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-background">
        <div
          className={`h-full rounded-full ${accent ? "bg-gold" : "bg-primary/80"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1.5 text-[10px] leading-snug text-muted-foreground">{hint}</p>
    </div>
  );
}

function WeightBar({
  label,
  value,
  className,
}: {
  label: string;
  value: number;
  className: string;
}) {
  const pct = Math.max(0, value) * 100;
  return (
    <div className="mt-2.5 flex items-center gap-3">
      <span className="w-24 text-xs text-muted-foreground">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-background">
        <div className={`h-full rounded-full ${className}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 text-right font-mono text-xs">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

const round4 = (v: number) => Math.round(v * 10000) / 10000;
