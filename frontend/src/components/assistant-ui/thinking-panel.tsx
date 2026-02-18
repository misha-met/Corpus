"use client";

/**
 * ThinkingPanel — wraps the assistant-ui Reasoning composable components
 * to display a collapsible "Thinking..." panel during RAG pipeline execution.
 *
 * - Defaults to closed (user can drop it down to see steps)
 * - Shows shimmer + "Thinking..." label while generating
 * - Shows static "Thinking" label when done
 * - Lists each pipeline step from AppContext.thinkingSteps as it arrives
 */

import { useAppState } from "@/context/app-context";
import { useAuiState } from "@assistant-ui/react";
import { type FC } from "react";
import {
  ReasoningRoot,
  ReasoningTrigger,
  ReasoningContent,
  ReasoningText,
} from "@/components/assistant-ui/reasoning";

export const ThinkingPanel: FC = () => {
  const { thinkingSteps } = useAppState();
  const isRunning = useAuiState((s) => s.message.status?.type === "running");
  // Flips true the moment the first text token lands and remains true
  // for the rest of the active stream.
  const hasText = useAuiState((s) =>
    s.message.parts.some((p) => p.type === "text"),
  );

  if (!isRunning && thinkingSteps.length === 0) return null;

  const triggerLabel = !isRunning
    ? "Done"
    : hasText
      ? "Generating answer..."
      : thinkingSteps.length > 0
        ? "Thinking..."
        : "Searching...";

  return (
    <ReasoningRoot variant="ghost" defaultOpen={false}>
      <ReasoningTrigger active={isRunning} label={triggerLabel} />
      <ReasoningContent aria-busy={isRunning}>
        <ReasoningText>
          <ul className="space-y-1.5">
            {thinkingSteps.map((step) => (
              <li key={step.id} className="flex items-start gap-2">
                <span className="mt-[7px] shrink-0 size-1 rounded-full bg-muted-foreground/40" />
                <span>{step.message}</span>
              </li>
            ))}
            {isRunning && (
              <li className="flex items-start gap-2 opacity-40">
                <span className="mt-[7px] shrink-0 size-1 rounded-full bg-muted-foreground/40 animate-pulse" />
                <span className="animate-pulse">working…</span>
              </li>
            )}
          </ul>
        </ReasoningText>
      </ReasoningContent>
    </ReasoningRoot>
  );
};
