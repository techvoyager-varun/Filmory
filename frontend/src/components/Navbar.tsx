import { useEffect, useState } from "react";
import { FilmoryLogo } from "@/components/FilmoryLogo";
import { Link, useNavigate } from "@tanstack/react-router";
import { LogOut, Menu, Search, User as UserIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useAuth } from "@/context/AuthContext";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/movies", label: "Movies" },
  { to: "/my-list", label: "My List" },
  { to: "/search", label: "Search" },
] as const;

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-colors duration-300",
        scrolled
          ? "border-b border-border bg-background/85 backdrop-blur-xl"
          : "bg-gradient-to-b from-background/90 to-transparent",
      )}
    >
      <nav className="mx-auto flex h-16 max-w-[1600px] items-center gap-6 px-4 md:px-10">
        <Link to="/" className="flex items-center font-display text-lg font-extrabold tracking-tight">
          <FilmoryLogo size={32} className="h-8 w-auto" />
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              activeOptions={{ exact: link.to === "/" }}
              activeProps={{ className: "text-foreground" }}
              inactiveProps={{ className: "text-muted-foreground" }}
              className="rounded-md px-3 py-1.5 text-sm font-medium transition-colors hover:text-foreground"
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button
            size="icon"
            variant="ghost"
            aria-label="Search movies"
            onClick={() => navigate({ to: "/search" })}
            className="rounded-md"
          >
            <Search className="size-4" aria-hidden />
          </Button>

          {isAuthenticated ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  className="hidden items-center gap-2 rounded-md px-2 md:inline-flex"
                  aria-label="Open profile menu"
                >
                  <span className="flex size-7 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                    {user?.name?.charAt(0).toUpperCase()}
                  </span>
                  <span className="max-w-24 truncate text-sm">{user?.name}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52 rounded-md">
                <DropdownMenuLabel className="truncate">{user?.email}</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link to="/profile">
                    <UserIcon className="size-4" aria-hidden /> Profile
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link to="/history">Watch history</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link to="/my-list">My List</Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => {
                    logout();
                    void navigate({ to: "/login" });
                  }}
                >
                  <LogOut className="size-4" aria-hidden /> Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div className="hidden items-center gap-2 md:flex">
              <Button asChild variant="ghost" size="sm" className="rounded-md">
                <Link to="/login">Login</Link>
              </Button>
              <Button asChild size="sm" className="rounded-md bg-primary hover:bg-primary-glow">
                <Link to="/register">Register</Link>
              </Button>
            </div>
          )}

          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
              <Button size="icon" variant="ghost" className="rounded-md md:hidden" aria-label="Open menu">
                <Menu className="size-5" aria-hidden />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-64 bg-surface">
              <SheetTitle className="px-4 pt-4 font-display">Filmory</SheetTitle>
              <div className="mt-4 flex flex-col gap-1 px-2">
                {[...NAV_LINKS, { to: "/history", label: "History" }, { to: "/profile", label: "Profile" }].map(
                  (link) => (
                    <Link
                      key={link.to}
                      to={link.to}
                      onClick={() => setOpen(false)}
                      className="rounded-md px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                      activeProps={{ className: "bg-muted text-foreground" }}
                      activeOptions={{ exact: link.to === "/" }}
                    >
                      {link.label}
                    </Link>
                  ),
                )}
                <div className="mt-3 border-t border-border pt-3">
                  {isAuthenticated ? (
                    <Button
                      variant="secondary"
                      className="w-full"
                      onClick={() => {
                        logout();
                        setOpen(false);
                        void navigate({ to: "/login" });
                      }}
                    >
                      Sign out
                    </Button>
                  ) : (
                    <div className="flex flex-col gap-2">
                      <Button asChild variant="secondary" onClick={() => setOpen(false)}>
                        <Link to="/login">Login</Link>
                      </Button>
                      <Button asChild className="bg-primary" onClick={() => setOpen(false)}>
                        <Link to="/register">Register</Link>
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </header>
  );
}
