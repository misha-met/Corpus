"use client";
import React, { lazy, Suspense } from "react";
import type { BackgroundTheme } from "@/context/theme-context";

const BACKGROUNDS: Record<BackgroundTheme, React.LazyExoticComponent<React.ComponentType<{ className?: string; paused?: boolean }>>> = {
  meteors: lazy(() => import("@/components/ui/meteors").then(m => ({ default: m.Meteors }))),
  rain: lazy(() => import("@/components/ui/rain").then(m => ({ default: m.RainBackground }))),
  mesh: lazy(() => import("@/components/ui/mesh-gradient").then(m => ({ default: m.MeshGradientBackground }))),
  starfield: lazy(() => import("@/components/ui/starfield").then(m => ({ default: m.StarfieldBackground }))),
  particles: lazy(() => import("@/components/ui/particles").then(m => ({ default: m.ParticleBackground }))),
  stars: lazy(() => import("@/components/ui/stars-background").then(m => ({ default: m.StarsBackground }))),
  darkveil: lazy(() => import("@/components/ui/dark-veil").then(m => ({ default: m.DarkVeilBackground }))),
};

export function BackgroundLayer({ theme, paused }: { theme: BackgroundTheme, paused?: boolean }) {
  const Bg = BACKGROUNDS[theme];
  // TODO: update background components to respect `paused` prop to skip their requestAnimationFrame loop
  return (
    <Suspense fallback={null}>
      <Bg className="absolute inset-0" paused={paused} />
    </Suspense>
  );
}
