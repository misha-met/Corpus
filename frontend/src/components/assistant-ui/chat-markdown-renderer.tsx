// ─── Phase 4 — ChatMarkdown renderer for assistant-ui ────────────────────────
// Thin glue component that satisfies the assistant-ui Text content-part slot.
//
// Pattern confirmed via Context7 (assistant-ui docs):
//   MessagePrimitive.Parts accepts components={{ Text: MyComponent }}
//   Inside the Text component, useMessagePartText() returns the raw message text.
//   Full React context hooks work freely inside the component since it renders
//   within the AssistantRuntimeProvider → AppProvider tree.
// ──────────────────────────────────────────────────────────────────────────────

"use client";

import type { FC } from "react";
import { useMessagePartText } from "@assistant-ui/react";
import { ChatMarkdown } from "@/components/chat-markdown";

/**
 * ChatMarkdownRenderer — registered as the `Text` component in
 * `MessagePrimitive.Parts` for assistant messages.
 *
 * Reads the current message text via the assistant-ui `useMessagePartText()`
 * hook and delegates all rendering (including citation link injection and
 * markdown formatting) to `ChatMarkdown`, which reads citations from
 * AppContext internally.
 */
export const ChatMarkdownRenderer: FC = () => {
  const { text } = useMessagePartText();
  return <ChatMarkdown content={text} />;
};
