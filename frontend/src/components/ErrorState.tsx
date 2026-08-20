import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function ErrorState({
  title = "Something went wrong",
  description = "We couldn't reach the recommendation service. Please try again.",
  onRetry,
  compact,
  className,
}: {
  title?: string;
  description?: string;
  onRetry?: (() => void) | undefined;
  compact?: boolean;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center rounded-none border border-dashed border-destructive/40 bg-destructive/5 px-6 text-center",
        compact ? "py-8" : "py-14",
        className,
      )}
    >
      <div className="mb-3 flex size-10 items-center justify-center rounded-none bg-destructive/15">
        <AlertTriangle className="size-5 text-destructive" aria-hidden />
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1.5 max-w-md text-sm text-muted-foreground">{description}</p>
      {onRetry ? (
        <Button
          variant="secondary"
          onClick={onRetry}
          className="mt-5 rounded-none border border-border bg-surface-raised/80 hover:bg-surface-raised"
        >
          <RotateCcw className="size-4" aria-hidden />
          Try again
        </Button>
      ) : null}
    </div>
  );
}
