import { useState } from "react";
import { FilmoryLogo } from "@/components/FilmoryLogo";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/context/AuthContext";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — Filmory" },
      { name: "description", content: "Sign in to Filmory to get personalised film recommendations." },
      { property: "og:title", content: "Sign in — Filmory" },
      { property: "og:description", content: "Access your Filmory watchlist, history and recommendations." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const { login, loginAsDemo } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState<"form" | "demo" | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error("Enter your email and password");
      return;
    }
    setPending("form");
    await login({ email, password });
    toast.success("Welcome back");
    void navigate({ to: "/" });
  };

  return (
    <AuthShell title="Welcome back" subtitle="Sign in to pick up where you left off.">
      <form onSubmit={submit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="bg-surface"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            className="bg-surface"
          />
        </div>

        <Button
          type="submit"
          disabled={pending !== null}
          className="w-full rounded-md bg-primary py-6 font-semibold hover:bg-primary-glow"
        >
          {pending === "form" ? <Loader2 className="size-4 animate-spin" aria-hidden /> : null}
          Login
        </Button>

        <Button
          type="button"
          variant="secondary"
          disabled={pending !== null}
          onClick={async () => {
            setPending("demo");
            await loginAsDemo();
            toast.success("Signed in as Demo Viewer");
            void navigate({ to: "/" });
          }}
          className="w-full rounded-md border border-border py-6"
        >
          {pending === "demo" ? <Loader2 className="size-4 animate-spin" aria-hidden /> : null}
          Continue as Demo User
        </Button>
      </form>

      <p className="mt-6 text-center text-sm text-muted-foreground">
        New here?{" "}
        <Link to="/register" className="font-semibold text-gold hover:underline">
          Create an account
        </Link>
      </p>
    </AuthShell>
  );
}

export function AuthShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="relative flex min-h-screen items-center justify-center px-4 py-28">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_0%,color-mix(in_oklab,var(--primary)_22%,transparent),transparent_70%)]"
      />
      <div className="relative w-full max-w-md rounded-xl border border-border bg-surface/80 p-8 shadow-2xl backdrop-blur-xl">
        <div className="mb-6 flex items-center">
          <FilmoryLogo size={32} className="h-8 w-auto" />
        </div>
        <h1 className="text-2xl font-bold">{title}</h1>
        <p className="mb-6 mt-1 text-sm text-muted-foreground">{subtitle}</p>
        {children}
      </div>
    </div>
  );
}
