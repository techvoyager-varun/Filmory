import { useState } from "react";
import { ChevronDown, ChevronUp, CheckCircle2, Cpu, FileText, Info } from "lucide-react";
import type { MovieEvidence } from "@/types/assistant";
import { cn } from "@/lib/utils";

interface EvidencePanelProps {
  evidence: MovieEvidence[];
  className?: string;
}

export function EvidencePanel({ evidence, className }: EvidencePanelProps) {
  const [expanded, setExpanded] = useState(false);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-border/70 bg-surface/50 text-xs backdrop-blur-sm transition-all",
        className,
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left font-medium text-muted-foreground transition hover:bg-surface hover:text-foreground"
      >
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-gold" />
          <span>Recommendation Evidence & AI Reasoning ({evidence.length} films analyzed)</span>
        </div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground/80">
          <span>{expanded ? "Hide Details" : "Inspect Why"}</span>
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </button>

      {expanded && (
        <div className="divide-y divide-border/50 border-t border-border/50 bg-surface/30 px-4 py-3">
          {evidence.map((item) => (
            <div key={item.movie_id} className="py-3 first:pt-1 last:pb-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-foreground text-sm">
                  {item.title} ({item.year})
                </span>
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1 rounded bg-surface-raised px-2 py-0.5 text-[10px] font-mono text-muted-foreground border border-border/60">
                    <Cpu className="h-3 w-3 text-gold" />
                    {item.damr_variant.toUpperCase()} • Score: {item.score.toFixed(3)}
                  </span>
                </div>
              </div>

              {/* Matched Constraints */}
              {item.matched_constraints && item.matched_constraints.length > 0 && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="text-muted-foreground font-medium">Matched criteria:</span>
                  {item.matched_constraints.map((c, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-400 border border-emerald-500/20"
                    >
                      <CheckCircle2 className="h-3 w-3" />
                      {c}
                    </span>
                  ))}
                </div>
              )}

              {/* Personalization Signals */}
              {item.personalization_signals && item.personalization_signals.length > 0 && (
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <span className="text-muted-foreground font-medium">Personalization:</span>
                  {item.personalization_signals.map((s, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center gap-1 rounded-md bg-primary/10 px-2 py-0.5 text-[11px] text-primary border border-primary/20"
                    >
                      <CheckCircle2 className="h-3 w-3" />
                      {s}
                    </span>
                  ))}
                </div>
              )}

              {/* Description Snippet */}
              {item.description_snippet && (
                <p className="mt-2 text-muted-foreground/80 leading-relaxed text-[11px]">
                  <FileText className="inline mr-1 h-3 w-3 opacity-60" />
                  {item.description_snippet}...
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
