import { createFileRoute } from "@tanstack/react-router";
import { RequireAuth } from "@/components/RequireAuth";
import { OnboardingWizard } from "@/components/onboarding/OnboardingWizard";

export const Route = createFileRoute("/onboarding")({
  head: () => ({
    meta: [
      { title: "Tell us what you like — Filmory" },
      {
        name: "description",
        content: "Pick your favourite genres and films so Filmory can recommend from day one.",
      },
      { property: "og:title", content: "Tell us what you like — Filmory" },
      {
        property: "og:description",
        content: "Cold-start preferences that seed your Filmory recommendations.",
      },
    ],
  }),
  component: () => (
    <RequireAuth>
      <OnboardingWizard />
    </RequireAuth>
  ),
});
