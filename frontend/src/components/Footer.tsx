import { Link } from "@tanstack/react-router";
import { FilmoryLogo } from "@/components/FilmoryLogo";

export function Footer() {
  return (
    <footer className="mt-16 border-t border-border bg-surface/60">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-10 md:flex-row md:items-center md:justify-between md:px-10">
        <div>
          <div className="flex items-center">
            <FilmoryLogo size={32} className="h-8 w-auto" />
          </div>
          <p className="mt-2 max-w-md text-sm text-muted-foreground">
            Personalised movie discovery, powered by your taste.
          </p>
        </div>
        <nav className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
          <Link to="/" className="hover:text-foreground">
            Home
          </Link>
          <Link to="/movies" className="hover:text-foreground">
            Movies
          </Link>
          <Link to="/search" className="hover:text-foreground">
            Search
          </Link>
          <Link to="/my-list" className="hover:text-foreground">
            My List
          </Link>
          <Link to="/history" className="hover:text-foreground">
            History
          </Link>
        </nav>
      </div>
    </footer>
  );
}
