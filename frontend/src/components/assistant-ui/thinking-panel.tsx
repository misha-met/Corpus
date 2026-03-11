"use client";

/**
 * ThinkingPanel — collapsible panel showing pipeline status steps and
 * streaming LLM reasoning tokens during RAG execution.
 *
 * - Status steps (from AppContext.thinkingSteps) listed as bullet items
 * - Live reasoning tokens (from AI SDK reasoning message parts) shown as streaming text
 * - Auto-opens when the first reasoning token arrives
 * - Shows shimmer label while generating; "Done" when finished
 */

import { useAppState } from "@/context/app-context";
import { ChatMarkdown } from "@/components/chat-markdown";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useEffect, useState } from "react";
import {
  ReasoningRoot,
  ReasoningTrigger,
  ReasoningContent,
  ReasoningText,
} from "@/components/assistant-ui/reasoning";

export const ThinkingPanel: FC = () => {
  const { thinkingSteps } = useAppState();
  const isRunning = useAuiState((s) => s.message.status?.type === "running");
  // Flips true the moment the first answer text token lands
  const hasText = useAuiState((s) =>
    s.message.parts.some((p) => p.type === "text"),
  );
  // Accumulate streaming reasoning tokens from AI SDK reasoning message parts
  const reasoningText = useAuiState((s) =>
    s.message.parts
      .filter((p) => p.type === "reasoning")
      .map((p) => {
        const part = p as { type: string; text?: string; reasoning?: string };
        return part.text ?? part.reasoning ?? "";
      })
      .join(""),
  );
  const hasReasoning = reasoningText.length > 0;

  // Controlled open state — auto-opens when reasoning tokens start streaming
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (isRunning && hasReasoning && !hasText) {
      setOpen(true);
    }
  }, [isRunning, hasReasoning, hasText]);

  if (!isRunning && thinkingSteps.length === 0 && !hasReasoning) return null;

  const active = isRunning && !hasText;
  const triggerLabel = !isRunning
    ? "Done"
    : hasText
      ? "Generating answer..."
      : hasReasoning || thinkingSteps.length > 0
        ? "Thinking..."
        : "Searching...";

  return (
    <ReasoningRoot variant="ghost" open={open} onOpenChange={setOpen}>
      <ReasoningTrigger active={active} label={triggerLabel} />
      <ReasoningContent aria-busy={isRunning} className="rounded-md bg-white/10 backdrop-blur-lg mt-1">
        <ReasoningText className="text-foreground/70">
          {/* Pipeline status steps */}
          {thinkingSteps.length > 0 && (
            <ul className="space-y-1.5">
              {thinkingSteps.map((step) => (
                <li key={step.id} className="flex items-start gap-2">
                  <span className="mt-1.75 shrink-0 size-1 rounded-full bg-muted-foreground/40" />
                  <span>{step.message}</span>
                </li>
              ))}
            </ul>
          )}

          {/* Streaming LLM reasoning content */}
          {hasReasoning && (
            <div className={thinkingSteps.length > 0 ? "mt-3 pt-3 border-t border-white/10" : ""}>
              <ChatMarkdown
                content={reasoningText}
                className="text-foreground/60 text-sm leading-relaxed [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2"
              />
              {active && <span className="animate-pulse ml-0.5 text-foreground/50">▍</span>}
            </div>
          )}

          {/* Spinner while nothing has arrived yet */}
          {isRunning && !hasReasoning && thinkingSteps.length === 0 && (
            <ul className="space-y-1.5">
              <li className="flex items-start gap-2 opacity-40">
                <span className="mt-1.75 shrink-0 size-1 rounded-full bg-muted-foreground/40 animate-pulse" />
                <span className="animate-pulse">working…</span>
              </li>
            </ul>
          )}
        </ReasoningText>
      </ReasoningContent>
    </ReasoningRoot>
  );
};
