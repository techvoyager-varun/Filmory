import { cn } from "@/lib/utils";

export function PosterSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "aspect-[2/3] w-[150px] shrink-0 animate-pulse rounded-none bg-surface-raised sm:w-[170px] lg:w-[190px]",
        className,
      )}
    />
  );
}

export function RowSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="flex gap-3 overflow-hidden px-4 md:px-10">
      {Array.from({ length: count }).map((_, i) => (
        <PosterSkeleton key={i} />
      ))}
    </div>
  );
}

export function GridSkeleton({ count = 12 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="aspect-[2/3] animate-pulse rounded-none bg-surface-raised" />
      ))}
    </div>
  );
}

export function HeroSkeleton() {
  return <div className="h-[62vh] w-full animate-pulse bg-surface md:h-[78vh]" />;
}
