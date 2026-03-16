"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMarkdown } from "@/components/chat-markdown";
import {
  ArrowUpIcon,
  BrainIcon,
  CheckIcon,
  CopyIcon,
  Loader2Icon,
  MicIcon,
  SquareIcon,
  ThumbsUpIcon,
  ThumbsDownIcon,
} from "lucide-react";
import {
  PickerRoot,
  PickerTrigger,
  PickerContent,
  PickerItem,
  pickerTriggerVariants,
} from "@/components/ui/picker";
import {
  ReasoningRoot,
  ReasoningTrigger,
  ReasoningContent,
  ReasoningText,
} from "@/components/assistant-ui/reasoning";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { cn } from "@/lib/utils";
import { TypewriterText } from "@/components/ui/typewriter-text";
import { useAppState } from "@/context/app-context";
import { getBackendBase } from "@/lib/backend-url";
import {
  saveSession,
  loadSession,
  deriveTitle,
  type FreeChatMessage,
} from "@/lib/session-store";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId(): string {
  return Math.random().toString(36).slice(2, 12);
}

const MODELS = [
  { id: "regular", name: "Regular", description: "Qwen3.5-35B-A3B" },
  { id: "deep-research", name: "Deep Research", description: "Qwen3.5-35B-A3B (Deep Retrieval + Thinking)" },
] as const;

const BACKEND_BASE = getBackendBase();

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/** Parse a single SSE text block into {event, data}. */
function parseSSEBlock(text: string): { event: string; data: string } | null {
  const lines = text.split(/\r?\n/);
  let evt = "message";
  let dat = "";
  for (const l of lines) {
    if (l.startsWith("event:")) evt = l.slice(6).trim();
    else if (l.startsWith("data:")) dat = l.slice(5).trimStart();
  }
  if (!dat) return null;
  return { event: evt, data: dat };
}

/** Yield parsed SSE events from a ReadableStream body. */
async function* parseSSEStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }, void, unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      const p = parseSSEBlock(block);
      if (p) yield p;
    }
  }
  if (buffer.trim()) {
    const p = parseSSEBlock(buffer.trim());
    if (p) yield p;
  }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

async function* freeformStreaming(
  messages: FreeChatMessage[],
  model: string,
  signal: AbortSignal,
  enableThinking: boolean = false,
  sessionId?: string,
): AsyncGenerator<{ event: string; data: string }, void, unknown> {
  const body = JSON.stringify({
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    model,
    enable_thinking: enableThinking,
    session_id: sessionId,
  });

  const res = await fetch(`${BACKEND_BASE}/api/freeform/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });

  if (!res.ok || !res.body) {
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      throw new Error("Backend not reachable — make sure the server is running on port 8000.");
    }
    if (res.status === 500) {
      throw new Error("Internal server error — something went wrong on the backend.");
    }
    throw new Error(`Freeform chat failed: HTTP ${res.status}`);
  }

  yield* parseSSEStream(res.body);
}

// ---------------------------------------------------------------------------
// AI title generation
// ---------------------------------------------------------------------------

/**
 * Ask the Qwen3.5-35B-A3B model for a 2-3 word title summarising the conversation.
 * Returns an empty string on any failure so the caller can fall back silently.
 */
async function generateTitleFromConversation(
  messages: FreeChatMessage[],
): Promise<string> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15_000);
  try {
    const titlePromptMessages = [
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      {
        role: "user" as const,
        content:
          "Give a 2-3 word title that summarises this conversation. Reply with ONLY the title, no punctuation, no explanation, nothing else.",
      },
    ];

    const res = await fetch(`${BACKEND_BASE}/api/freeform/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: titlePromptMessages, model: "regular", enable_thinking: false }),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) return "";

    let fullText = "";

    for await (const { event, data } of parseSSEStream(res.body)) {
      if (event === "token" && data) {
        try {
          const parsed = JSON.parse(data);
          if (typeof parsed.text === "string") fullText += parsed.text;
        } catch { /* skip */ }
      } else if (event === "done") {
        break;
      }
    }

    // Strip quotes, asterisks, markdown, newlines; trim; cap at 50 chars
    return fullText.replace(/["'*#\n\r]/g, "").trim().slice(0, 50) || "";
  } catch {
    return "";
  } finally {
    clearTimeout(timeoutId);
  }
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface FreeformChatPanelProps {
  restoredSessionId?: string | null;
  restoredMessages?: FreeChatMessage[] | null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FreeformChatPanel({
  restoredSessionId,
  restoredMessages,
}: FreeformChatPanelProps) {
  const { chatMode } = useAppState();
  const [messages, setMessages] = useState<FreeChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbackState, setFeedbackState] = useState<Record<string, "up" | "down" | null>>({});
  const [selectedModel, setSelectedModel] = useState("regular");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);

  const viewportRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingTextRef = useRef("");
  const updateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const messagesRef = useRef<FreeChatMessage[]>([]);
  const sessionIdRef = useRef<string>(crypto.randomUUID());
  const isRestoredRef = useRef(false);
  const titleGeneratedRef = useRef(false);
  const thinkingTextRef = useRef("");

  // ── Sync refs ──────────────────────────────────────────────────────────────
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Reset thinking mode when switching to a model that doesn't support it
  useEffect(() => {
    if (selectedModel === "deep-research") setThinkingEnabled(false);
  }, [selectedModel]);

  // Scroll to bottom on new messages
  useEffect(() => {
    viewportRef.current?.scrollTo({ top: viewportRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // Focus textarea on mount + reset scrollTop so placeholder is never clipped
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.scrollTop = 0;
    el.focus();
  }, []);

  // ── Auto-resize textarea ───────────────────────────────────────────────────
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`;
    el.scrollTop = 0;
  }, []);

  useEffect(() => { resizeTextarea(); }, [inputValue, resizeTextarea]);

  // Re-run resize when the panel transitions from hidden → visible.
  // The panel is always mounted but toggled via display:none ("hidden" class),
  // so scrollHeight = 0 while hidden — ResizeObserver fires when it reappears.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => { resizeTextarea(); });
    ro.observe(el);
    return () => ro.disconnect();
  }, [resizeTextarea]);

  // ── Restore session ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!restoredSessionId) return;
    if (restoredMessages && restoredMessages.length > 0) {
      sessionIdRef.current = restoredSessionId;
      isRestoredRef.current = true;
      setMessages(restoredMessages);
      return;
    }
    loadSession(restoredSessionId).then((session) => {
      if (session && session.mode === "freeform" && session.messages.length > 0) {
        sessionIdRef.current = restoredSessionId;
        isRestoredRef.current = true;
        setMessages(session.messages);
      }
    }).catch(console.error);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restoredSessionId]);

  // ── Session persistence ────────────────────────────────────────────────────
  const saveCurrentSession = useCallback((msgs: FreeChatMessage[]) => {
    if (msgs.length === 0) return;
    const now = Date.now();
    saveSession({
      id: sessionIdRef.current,
      mode: "freeform",
      title: deriveTitle(msgs),
      messages: msgs,
      createdAt: now,
      updatedAt: now,
    }).catch(console.error);

    // After the first complete exchange, generate an AI 2-3 word title once
    const hasUser = msgs.some((m) => m.role === "user");
    const hasAssistant = msgs.some(
      (m) => m.role === "assistant" && m.content.trim().length > 0,
    );
    if (hasUser && hasAssistant && !titleGeneratedRef.current) {
      titleGeneratedRef.current = true;
      const sessionId = sessionIdRef.current;
      generateTitleFromConversation(msgs)
        .then((aiTitle) => {
          if (!aiTitle) return;
          loadSession(sessionId)
            .then((existing) => {
              if (!existing) return;
              saveSession({ ...existing, title: aiTitle }).catch(console.error);
            })
            .catch(console.error);
        })
        .catch(console.error);
    }
  }, []);

  const startNewConversation = useCallback(() => {
    setMessages([]);
    setErrorText("");
    sessionIdRef.current = crypto.randomUUID();
    isRestoredRef.current = false;
    titleGeneratedRef.current = false;
    setInputValue("");
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  // ── Speech-to-text ─────────────────────────────────────────────────────────
  const applyTranscript = useCallback(
    (chunk: string, _isFinal: boolean) => {
      if (!chunk) return;
      const el = textareaRef.current;
      setInputValue((prev) => {
        const pos = el?.selectionStart ?? prev.length;
        const sep = pos > 0 && !prev.slice(0, pos).endsWith(" ") ? " " : "";
        const next = prev.slice(0, pos) + sep + chunk + prev.slice(pos);
        const newPos = pos + sep.length + chunk.length;
        requestAnimationFrame(() => {
          el?.setSelectionRange(newPos, newPos);
          el?.focus();
        });
        return next;
      });
    },
    [],
  );

  const { status: sttStatus, isListening, toggle: toggleMic, stop: stopMic } = useSpeechToText({
    onTranscript: applyTranscript,
    onPermissionDenied: () => {},
    onNoSpeech: () => {},
    onError: () => {},
  });

  const isTranscribing = sttStatus === "transcribing";

  useEffect(() => {
    if (isStreaming && (isListening || isTranscribing)) stopMic();
  }, [isStreaming, isListening, isTranscribing, stopMic]);

  // ── Stop streaming ─────────────────────────────────────────────────────────
  const stop = useCallback(() => { abortRef.current?.abort(); }, []);

  // ── Send message ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(
    async (text: string) => {
      setErrorText("");
      const userMsg: FreeChatMessage = {
        id: generateId(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      const assistantId = generateId();
      const assistantMsg: FreeChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      };

      const nextMessages = [...messagesRef.current, userMsg, assistantMsg];
      streamingTextRef.current = "";
      thinkingTextRef.current = "";
      setMessages(nextMessages);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const scheduleUpdate = () => {
        if (updateTimerRef.current) return;
        updateTimerRef.current = setTimeout(() => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: streamingTextRef.current,
                    thinkingContent: thinkingTextRef.current || undefined,
                    timestamp: Date.now(),
                  }
                : m,
            ),
          );
          updateTimerRef.current = null;
        }, 50);
      };

      const historyToSend = [...messagesRef.current, userMsg];

      try {
        for await (const { event, data } of freeformStreaming(
          historyToSend,
          selectedModel,
          controller.signal,
          thinkingEnabled,
          sessionIdRef.current,
        )) {
          if (event === "trace_id") {
            try {
              const parsed = JSON.parse(data);
              if (parsed.trace_id) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, traceId: parsed.trace_id, spanId: parsed.span_id || "" }
                      : m
                  ),
                );
              }
            } catch { /* skip */ }
          } else if (event === "thinking_token") {
            try {
              const parsed = JSON.parse(data);
              const token = typeof parsed.text === "string" ? parsed.text : "";
              if (token) {
                thinkingTextRef.current += token;
                scheduleUpdate();
              }
            } catch { /* skip */ }
          } else if (event === "token") {
            try {
              const parsed = JSON.parse(data);
              const token = typeof parsed.text === "string" ? parsed.text : "";
              if (token) {
                streamingTextRef.current += token;
                scheduleUpdate();
              }
            } catch { /* skip */ }
          } else if (event === "error") {
            try { setErrorText(JSON.parse(data).error ?? "Streaming error"); }
            catch { setErrorText("Streaming error"); }
          } else if (event === "complete") {
            if (updateTimerRef.current) {
              clearTimeout(updateTimerRef.current);
              updateTimerRef.current = null;
            }
            const finalText = streamingTextRef.current;
            setMessages((prev) => {
              const final = prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: finalText,
                      thinkingContent: thinkingTextRef.current || undefined,
                      timestamp: Date.now(),
                    }
                  : m,
              );
              saveCurrentSession(final);
              return final;
            });
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name !== "AbortError") {
          const isNetworkErr = err instanceof TypeError &&
            (err.message === "Failed to fetch" || err.message.includes("fetch failed") || err.message.includes("ECONNREFUSED"));
          setErrorText(
            isNetworkErr
              ? "Backend not reachable — make sure the server is running on port 8000."
              : (err.message || "An unexpected error occurred")
          );
        }
        if ((err as Error)?.name === "AbortError") {
          setMessages((prev) => prev.filter((m) => m.id !== assistantId));
        }
      } finally {
        if (updateTimerRef.current) {
          clearTimeout(updateTimerRef.current);
          updateTimerRef.current = null;
        }
        abortRef.current = null;
        setIsStreaming(false);
      }
    },
    [saveCurrentSession, selectedModel, thinkingEnabled],
  );

  // ── Form submit ────────────────────────────────────────────────────────────
  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = inputValue.trim();
    if (!text || isStreaming) return;
    setInputValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    sendMessage(text);
  }

  async function copyToClipboard(id: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch { /* clipboard not available */ }
  }

  const isEmpty = !inputValue.trim();

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className="@container flex h-full flex-col"
      style={{ ["--thread-max-width" as string]: "45rem" }}
    >
      {/* Scrollable viewport */}
      <div
        ref={viewportRef}
        className="relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll scroll-smooth px-4 pt-4"
      >
        {/* Welcome screen */}
        {messages.length === 0 && (
          <div className="mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
            <div className="flex w-full grow flex-col items-center justify-center">
              <div className="flex size-full flex-col justify-center px-4">
                <TypewriterText
                  key={chatMode}
                  text="General chat"
                  typingSpeed={80}
                  className="font-semibold text-2xl"
                />
                <p className="fade-in slide-in-from-bottom-1 animate-in fill-mode-both text-muted-foreground text-xl delay-75 duration-200">
                  Answers from model knowledge only
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((message) => {
          if (!message.content && !message.thinkingContent && message.role === "assistant" && !isStreaming) return null;
          const isUser = message.role === "user";
          // Show placeholder spinner only when there's no content of any kind yet
          const isPlaceholder = message.role === "assistant" && !message.content && !message.thinkingContent && isStreaming;
          // Thinking is actively streaming when we have thinking content but no answer tokens yet
          const isThinkingStreaming = isStreaming && !!message.thinkingContent && !message.content;

          if (isUser) {
            return (
              <div
                key={message.id}
                className="fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-4 duration-150 *:col-start-2"
                data-role="user"
              >
                <div className="relative col-start-2 min-w-0">
                  <div
                    className="wrap-break-word px-4 py-3 text-white text-sm bg-white/10 backdrop-blur-lg"
                    style={{ borderRadius: "18px 18px 4px 18px" }}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              </div>
            );
          }

          return (
            <div
              key={message.id}
              className="fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-4 duration-150"
              data-role="assistant"
            >
              {/* Loading spinner — before any tokens arrive */}
              {isPlaceholder && (
                <span className="px-2 text-muted-foreground animate-pulse">Loading model…</span>
              )}

              {/* Reasoning panel — rendered naked, no bubble */}
              {message.thinkingContent && (
                <ReasoningRoot defaultOpen variant="ghost" className="mb-2">
                  <ReasoningTrigger active={isThinkingStreaming} />
                  <ReasoningContent aria-busy={isThinkingStreaming} className="rounded-md bg-white/10 backdrop-blur-lg mt-1">
                    <ReasoningText className="text-foreground/70">
                      <ChatMarkdown content={message.thinkingContent} citations={[]} />
                    </ReasoningText>
                  </ReasoningContent>
                </ReasoningRoot>
              )}

              {/* Answer bubble — only rendered once answer tokens arrive */}
              {message.content && (
                <div
                  className="wrap-break-word w-fit max-w-full px-4 py-3 text-foreground text-base leading-[1.65] bg-white/10 backdrop-blur-lg"
                  style={{ borderRadius: "18px 18px 18px 4px" }}
                >
                  <ChatMarkdown content={message.content} citations={[]} />
                </div>
              )}

              {!isPlaceholder && message.content && (
                <div className="mt-1 ml-2 flex gap-0.5">
                  <button
                    type="button"
                    onClick={() => copyToClipboard(message.id, message.content)}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                    title="Copy"
                  >
                    {copiedId === message.id ? (
                      <CheckIcon className="size-3.5" />
                    ) : (
                      <CopyIcon className="size-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const newState = feedbackState[message.id] === "up" ? null : "up";
                      setFeedbackState((prev) => ({ ...prev, [message.id]: newState }));
                      if (newState && message.traceId) {
                        fetch(`${BACKEND_BASE}/api/feedback`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            span_id: message.spanId,
                            trace_id: message.traceId,
                            label: "👍",
                            score: 1.0,
                          }),
                        }).catch(() => {});
                      }
                    }}
                    className={cn(
                      "inline-flex items-center rounded-md px-2 py-1 text-xs transition-colors",
                      feedbackState[message.id] === "up"
                        ? "text-green-400 bg-green-400/10"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                    title="Good response"
                  >
                    <ThumbsUpIcon className="size-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const newState = feedbackState[message.id] === "down" ? null : "down";
                      setFeedbackState((prev) => ({ ...prev, [message.id]: newState }));
                      if (newState && message.traceId) {
                        fetch(`${BACKEND_BASE}/api/feedback`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            span_id: message.spanId,
                            trace_id: message.traceId,
                            label: "👎",
                            score: 0.0,
                          }),
                        }).catch(() => {});
                      }
                    }}
                    className={cn(
                      "inline-flex items-center rounded-md px-2 py-1 text-xs transition-colors",
                      feedbackState[message.id] === "down"
                        ? "text-red-400 bg-red-400/10"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                    title="Poor response"
                  >
                    <ThumbsDownIcon className="size-3.5" />
                  </button>
                </div>
              )}
            </div>
          );
        })}

        {/* Error */}
        {errorText && (
          <div className="mx-auto w-full max-w-(--thread-max-width) px-2 pb-2">
            <div className="rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm">
              {errorText}
            </div>
          </div>
        )}

        {/* Sticky footer with composer */}
        <div className="sticky bottom-0 mx-auto mt-auto flex w-full max-w-(--thread-max-width) flex-col gap-3 overflow-visible bg-transparent pb-4 pt-3 md:pb-6">
          {messages.length > 0 && (
            <div className="flex justify-center">
              <button
                type="button"
                onClick={startNewConversation}
                className="text-xs text-muted-foreground hover:text-foreground hover:bg-accent px-3 py-1 rounded-md transition-colors"
              >
                New conversation
              </button>
            </div>
          )}

          {/* Composer pill — identical to RAG Thread */}
          <form onSubmit={onSubmit}>
            <div
              className="flex w-full items-end gap-2 rounded-3xl px-4 py-1 outline-none transition-shadow"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.12)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)",
                backdropFilter: "blur(12px)",
              }}
            >
              {/* Textarea */}
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit(e as unknown as React.FormEvent);
                  }
                }}
                placeholder="What do you want to know?"
                disabled={isStreaming}
                rows={1}
                className="max-h-40 flex-1 resize-none bg-transparent py-2 text-sm text-white outline-none placeholder:text-[#555555] focus-visible:ring-0 disabled:opacity-50 overflow-y-auto"
              />

              {/* Thinking mode toggle — only shown for the 35B model */}
              {selectedModel === "regular" && (
                <button
                  type="button"
                  onClick={() => setThinkingEnabled((v) => !v)}
                  aria-pressed={thinkingEnabled}
                  title={thinkingEnabled ? "Thinking mode on — click to disable" : "Enable thinking mode"}
                  className={cn(
                    pickerTriggerVariants({ variant: "ghost", size: "sm" }),
                    "shrink-0",
                    thinkingEnabled ? "text-blue-400" : "text-muted-foreground",
                  )}
                >
                  {thinkingEnabled ? (
                    <span className="flex items-center gap-1.5">
                      <span className="size-1.5 rounded-full bg-blue-400 shrink-0" />
                      <span className="font-medium">Think</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5">
                      <BrainIcon className="size-3.5" />
                      <span className="font-medium">Think</span>
                    </span>
                  )}
                </button>
              )}

              {/* Model selector */}
              <PickerRoot open={pickerOpen} onOpenChange={setPickerOpen}>
                <PickerTrigger
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground"
                  aria-label="Select model"
                >
                  <span className="font-medium">
                    {MODELS.find((m) => m.id === selectedModel)?.name ?? selectedModel}
                  </span>
                </PickerTrigger>
                <PickerContent className="min-w-44" align="end">
                  {MODELS.map((m) => (
                    <PickerItem
                      key={m.id}
                      selected={m.id === selectedModel}
                      description={m.description}
                      onClick={() => { setSelectedModel(m.id); setPickerOpen(false); }}
                    >
                      {m.name}
                    </PickerItem>
                  ))}
                </PickerContent>
              </PickerRoot>

              {/* White circle button: mic / send / stop */}
              <div className="relative shrink-0 size-9 rounded-full bg-white">
                {/* Mic — shown when empty and not streaming */}
                <button
                  type="button"
                  aria-label={isTranscribing ? "Transcribing…" : isListening ? "Stop listening" : "Voice input"}
                  onClick={() => { if (!isStreaming && !isTranscribing) toggleMic(); }}
                  className={cn(
                    "absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150",
                    (!isEmpty || isStreaming) ? "scale-0 opacity-0 pointer-events-none" : "",
                    isListening ? "bg-red-500" : "",
                  )}
                >
                  {isTranscribing ? (
                    <Loader2Icon className="size-4 text-black animate-spin" />
                  ) : (
                    <MicIcon className={cn("size-4 text-black", isListening ? "animate-pulse" : "")} />
                  )}
                </button>

                {/* Send — shown when text present and not streaming */}
                <button
                  type="submit"
                  aria-label="Send message"
                  className={cn(
                    "absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150",
                    (isEmpty || isStreaming) ? "scale-0 opacity-0 pointer-events-none" : "",
                  )}
                >
                  <ArrowUpIcon className="size-4 text-black" />
                </button>

                {/* Stop — shown when streaming */}
                <button
                  type="button"
                  aria-label="Stop generating"
                  onClick={stop}
                  className={cn(
                    "absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150",
                    !isStreaming ? "scale-0 opacity-0 pointer-events-none" : "",
                  )}
                >
                  <SquareIcon className="size-3 fill-black text-black" />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
