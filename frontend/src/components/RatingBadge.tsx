import { Star } from "lucide-react";
import { cn } from "@/lib/utils";

export function RatingBadge({ rating, className }: { rating: number; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md bg-background/70 px-2 py-0.5 text-xs font-semibold text-gold backdrop-blur",
        className,
      )}
    >
      <Star className="size-3 fill-current" aria-hidden />
      {rating.toFixed(1)}
    </span>
  );
}

export function MatchBadge({ score, className }: { score: number; className?: string }) {
  const percent = Math.round(score * 100);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md bg-primary/90 px-2 py-0.5 text-xs font-bold text-primary-foreground",
        className,
      )}
    >
      {percent}% Match
    </span>
  );
}
