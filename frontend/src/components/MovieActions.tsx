import { Check, Heart, Play, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useUserData } from "@/context/UserDataContext";
import { cn } from "@/lib/utils";
import type { Movie } from "@/types/movie";

type Size = "sm" | "lg" | "icon";

export function PlayButton({ movie, size = "lg" }: { movie: Movie; size?: Size }) {
  const { play } = useUserData();

  if (size === "icon") {
    return (
      <Button
        size="icon"
        aria-label={`Play ${movie.title}`}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          void play(movie);
        }}
        className="size-9 rounded-none bg-foreground text-background hover:bg-foreground/85"
      >
        <Play className="size-4 fill-current" aria-hidden />
      </Button>
    );
  }

  return (
    <Button
      size={size === "lg" ? "lg" : "default"}
      onClick={() => void play(movie)}
      className="glow-primary rounded-none h-10 px-4 text-sm md:h-11 md:px-6 md:text-base bg-primary font-semibold text-primary-foreground hover:bg-primary-glow"
    >
      <Play className="size-4 fill-current" aria-hidden />
      Play
    </Button>
  );
}

export function LikeButton({ movie, size = "lg" }: { movie: Movie; size?: Size }) {
  const { isLiked, toggleLike } = useUserData();
  const liked = isLiked(movie.movieId);

  return (
    <Button
      variant="secondary"
      size={size === "icon" ? "icon" : size === "lg" ? "lg" : "default"}
      aria-pressed={liked}
      aria-label={liked ? `Unlike ${movie.title}` : `Like ${movie.title}`}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        void toggleLike(movie);
      }}
      className={cn(
        "rounded-none border border-border bg-surface-raised/80 hover:bg-surface-raised",
        size !== "icon" &&' h-10 px-4 text-sm md:h-11 md:px-6 md:text-base',
        size === "icon" && "size-9",
      )}
    >
      <Heart className={cn("size-4", liked && "fill-primary text-primary")} aria-hidden />
      {size !== "icon" ? (liked ? "Liked" : "Like") : null}
    </Button>
  );
}

export function MyListButton({ movie, size = "lg" }: { movie: Movie; size?: Size }) {
  const { isInList, toggleMyList } = useUserData();
  const inList = isInList(movie.movieId);

  return (
    <Button
      variant="secondary"
      size={size === "icon" ? "icon" : size === "lg" ? "lg" : "default"}
      aria-pressed={inList}
      aria-label={inList ? `Remove ${movie.title} from My List` : `Add ${movie.title} to My List`}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        void toggleMyList(movie);
      }}
      className={cn(
        "rounded-none border border-border bg-surface-raised/80 hover:bg-surface-raised",
        size !== "icon" &&' h-10 px-4 text-sm md:h-11 md:px-6 md:text-base',
        size === "icon" && "size-9",
      )}
    >
      {inList ? <Check className="size-4 text-gold" aria-hidden /> : <Plus className="size-4" aria-hidden />}
      {size !== "icon" ? (inList ? "In My List" : "My List") : null}
    </Button>
  );
}
