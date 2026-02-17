"use client";

import { TooltipProvider } from "@/components/ui/tooltip";
import { AppProvider } from "@/context/app-context";
import type { ReactNode } from "react";

/**
 * Client-side root providers.
 * Wraps the app in any providers that require the "use client" directive.
 * AppProvider (Phase 3) wraps the tree so any client component can access
 * AppState and dispatch without prop-drilling.
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <AppProvider>
      <TooltipProvider>{children}</TooltipProvider>
    </AppProvider>
  );
}
