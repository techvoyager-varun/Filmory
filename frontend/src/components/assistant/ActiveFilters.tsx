import { Filter, Search, Heart, Tag, Clock, Star, Calendar, Ban, Film, X } from "lucide-react";
import type { MovieIntent } from "@/types/assistant";
import { cn } from "@/lib/utils";

interface ActiveFiltersProps {
  intent: MovieIntent | null | undefined;
  sessionId?: string | null | undefined;
  onFilterRemoved?: ((updatedIntent: MovieIntent) => void) | undefined;
  className?: string | undefined;
}

export function ActiveFilters({ intent, sessionId, onFilterRemoved, className }: ActiveFiltersProps) {
  if (!intent) return null;

  const hasAnyFilter =
    Boolean(intent.semantic_query) ||
    intent.preferred_genres.length > 0 ||
    intent.excluded_genres.length > 0 ||
    intent.min_year !== null ||
    intent.max_year !== null ||
    intent.max_runtime_minutes !== null ||
    intent.min_rating !== null ||
    intent.reference_titles.length > 0 ||
    Boolean(intent.mood);

  if (!hasAnyFilter) return null;

  const handleRemoveFilter = async (filterKey: string) => {
    if (!sessionId) return;
    try {
      const { clearSessionFilter } = await import("@/api/assistant");
      const updated = await clearSessionFilter(sessionId, filterKey);
      onFilterRemoved?.(updated);
    } catch (err) {
      console.error("Failed to clear filter:", err);
    }
  };

  const RemoveButton = ({ filterKey }: { filterKey: string }) => (
    <button
      onClick={(e) => {
        e.stopPropagation();
        handleRemoveFilter(filterKey);
      }}
      className="ml-0.5 rounded-full p-0.5 transition-colors hover:bg-white/10 focus:outline-none focus:ring-1 focus:ring-gold/50"
      aria-label={`Remove ${filterKey} filter`}
      title="Remove filter"
    >
      <X className="h-2.5 w-2.5 opacity-60 hover:opacity-100" />
    </button>
  );

  return (
    <div
      className={cn(
        "rounded-xl border border-border/80 bg-surface/80 p-3.5 backdrop-blur-md shadow-sm transition-all",
        className,
      )}
    >
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <Filter className="h-3.5 w-3.5 text-gold" />
          <span>Active Assistant Intent & Filters</span>
        </div>
        {sessionId && (
          <button
            onClick={() => handleRemoveFilter("all")}
            className="text-[10px] font-medium text-muted-foreground/70 hover:text-red-400 transition-colors px-1.5 py-0.5 rounded hover:bg-red-500/10"
            title="Clear all filters"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        {intent.semantic_query ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-gold/30 bg-gold/10 px-2.5 py-1 text-gold">
            <Search className="h-3 w-3" />
            <span className="font-medium">"{intent.semantic_query}"</span>
            {sessionId && <RemoveButton filterKey="semantic_query" />}
          </span>
        ) : null}

        {intent.mood ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/10 px-2.5 py-1 text-primary">
            <Heart className="h-3 w-3" />
            <span>Mood: {intent.mood}</span>
            {sessionId && <RemoveButton filterKey="mood" />}
          </span>
        ) : null}

        {intent.preferred_genres.map((g) => (
          <span
            key={g}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-raised px-2.5 py-1 text-foreground"
          >
            <Tag className="h-3 w-3 text-emerald-400" />
            <span>{g}</span>
            {sessionId && <RemoveButton filterKey={`preferred_genres:${g}`} />}
          </span>
        ))}

        {intent.excluded_genres.map((g) => (
          <span
            key={g}
            className="inline-flex items-center gap-1 rounded-full border border-red-500/30 bg-red-500/10 px-2.5 py-1 text-red-400"
          >
            <Ban className="h-3 w-3" />
            <span>No {g}</span>
            {sessionId && <RemoveButton filterKey={`excluded_genres:${g}`} />}
          </span>
        ))}

        {intent.min_year !== null || intent.max_year !== null ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-raised px-2.5 py-1 text-muted-foreground">
            <Calendar className="h-3 w-3 text-sky-400" />
            <span>
              {intent.min_year ?? "Any"} – {intent.max_year ?? "Present"}
            </span>
            {sessionId && (
              <>
                {intent.min_year !== null && <RemoveButton filterKey="min_year" />}
                {intent.max_year !== null && <RemoveButton filterKey="max_year" />}
              </>
            )}
          </span>
        ) : null}

        {intent.max_runtime_minutes !== null ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-raised px-2.5 py-1 text-muted-foreground">
            <Clock className="h-3 w-3 text-amber-400" />
            <span>≤ {intent.max_runtime_minutes} min</span>
            {sessionId && <RemoveButton filterKey="max_runtime_minutes" />}
          </span>
        ) : null}

        {intent.min_rating !== null ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-raised px-2.5 py-1 text-muted-foreground">
            <Star className="h-3 w-3 text-yellow-400 fill-yellow-400/20" />
            <span>★ ≥ {intent.min_rating.toFixed(1)}</span>
            {sessionId && <RemoveButton filterKey="min_rating" />}
          </span>
        ) : null}

        {intent.reference_titles.map((t) => (
          <span
            key={t}
            className="inline-flex items-center gap-1 rounded-full border border-purple-500/30 bg-purple-500/10 px-2.5 py-1 text-purple-300"
          >
            <Film className="h-3 w-3" />
            <span>Like "{t}"</span>
            {sessionId && <RemoveButton filterKey="reference_titles" />}
          </span>
        ))}
      </div>
    </div>
  );
}
