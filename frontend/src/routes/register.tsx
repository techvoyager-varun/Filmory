import { useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthShell } from "@/routes/login";
import { useAuth } from "@/context/AuthContext";

export const Route = createFileRoute("/register")({
  head: () => ({
    meta: [
      { title: "Create your account — Filmory" },
      { name: "description", content: "Create a Filmory account and tell us what you like to watch." },
      { property: "og:title", content: "Create your account — Filmory" },
      { property: "og:description", content: "Join Filmory and get recommendations tuned to your taste." },
    ],
  }),
  component: RegisterPage,
});

function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [pending, setPending] = useState(false);

  const set = (key: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.password) {
      toast.error("Please fill in every field");
      return;
    }
    if (form.password !== form.confirm) {
      toast.error("Passwords don't match");
      return;
    }
    setPending(true);
    await register({ name: form.name, email: form.email, password: form.password });
    toast.success(`Welcome, ${form.name}`);
    void navigate({ to: "/onboarding" });
  };

  return (
    <AuthShell title="Create your account" subtitle="A few details and you're in.">
      <form onSubmit={submit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" value={form.name} onChange={set("name")} placeholder="Ada Lovelace" className="bg-surface" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            value={form.email}
            onChange={set("email")}
            placeholder="you@example.com"
            className="bg-surface"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            value={form.password}
            onChange={set("password")}
            placeholder="••••••••"
            className="bg-surface"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="confirm">Confirm password</Label>
          <Input
            id="confirm"
            type="password"
            value={form.confirm}
            onChange={set("confirm")}
            placeholder="••••••••"
            className="bg-surface"
          />
        </div>

        <Button
          type="submit"
          disabled={pending}
          className="w-full rounded-md bg-primary py-6 font-semibold hover:bg-primary-glow"
        >
          {pending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : null}
          Create account
        </Button>
      </form>

      <p className="mt-6 text-center text-sm text-muted-foreground">
        Already have an account?{" "}
        <Link to="/login" className="font-semibold text-gold hover:underline">
          Sign in
        </Link>
      </p>
    </AuthShell>
  );
}
