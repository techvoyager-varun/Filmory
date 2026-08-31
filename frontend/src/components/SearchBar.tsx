import { Search, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export function SearchBar({
  value,
  onChange,
  placeholder = "Search movies...",
  autoFocus,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  return (
    <div className="relative">
      <Search
        className="pointer-events-none absolute left-4 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        aria-hidden
      />
      <Input
        type="search"
        value={value}
        autoFocus={autoFocus}
        aria-label="Search movies"
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="h-13 rounded-lg border-border bg-surface/80 pl-11 pr-11 text-base backdrop-blur placeholder:text-muted-foreground focus-visible:ring-primary"
      />
      {value ? (
        <Button
          size="icon"
          variant="ghost"
          aria-label="Clear search"
          onClick={() => onChange("")}
          className="absolute right-2 top-1/2 size-8 -translate-y-1/2 rounded-md"
        >
          <X className="size-4" aria-hidden />
        </Button>
      ) : null}
    </div>
  );
}
