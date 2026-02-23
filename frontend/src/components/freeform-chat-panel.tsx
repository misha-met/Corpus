"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMarkdown } from "@/components/chat-markdown";
import * as SelectPrimitive from "@radix-ui/react-select";
import {
  ArrowUpIcon,
  CheckIcon,
  ChevronDownIcon,
  CopyIcon,
  MicIcon,
  SquareIcon,
} from "lucide-react";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { cn } from "@/lib/utils";
import { TypewriterText } from "@/components/ui/typewriter-text";
import { useAppState } from "@/context/app-context";
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
  { id: "regular", name: "Regular", description: "Qwen3-30B" },
  { id: "deep-research", name: "Deep Research", description: "Qwen3-80B (64GB+)" },
] as const;

const BACKEND_BASE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) ||
  "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

async function* freeformStreaming(
  messages: FreeChatMessage[],
  model: string,
  signal: AbortSignal,
): AsyncGenerator<{ event: string; data: string }, void, unknown> {
  const body = JSON.stringify({
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    model,
  });

  const res = await fetch(`${BACKEND_BASE}/api/freeform/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Freeform chat failed: HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const parseBlock = (text: string): { event: string; data: string } | null => {
    const lines = text.split(/\r?\n/);
    let evt = "message";
    let dat = "";
    for (const l of lines) {
      if (l.startsWith("event:")) evt = l.slice(6).trim();
      else if (l.startsWith("data:")) dat = l.slice(5).trimStart();
    }
    if (!dat) return null;
    return { event: evt, data: dat };
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      const p = parseBlock(block);
      if (p) yield p;
    }
  }
  if (buffer.trim()) {
    const p = parseBlock(buffer.trim());
    if (p) yield p;
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
  const [selectedModel, setSelectedModel] = useState("regular");

  const viewportRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingTextRef = useRef("");
  const updateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const messagesRef = useRef<FreeChatMessage[]>([]);
  const sessionIdRef = useRef<string>(crypto.randomUUID());
  const isRestoredRef = useRef(false);
  const lastInsertedRef = useRef("");
  const lastChunkRef = useRef<{ text: string; at: number } | null>(null);

  // ── Sync refs ──────────────────────────────────────────────────────────────
  useEffect(() => { messagesRef.current = messages; }, [messages]);

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
  }, []);

  const startNewConversation = useCallback(() => {
    setMessages([]);
    setErrorText("");
    sessionIdRef.current = crypto.randomUUID();
    isRestoredRef.current = false;
    setInputValue("");
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  // ── Speech-to-text ─────────────────────────────────────────────────────────
  const applyTranscript = useCallback(
    (chunk: string, isFinal: boolean) => {
      if (isFinal && !chunk) {
        lastChunkRef.current = null;
        lastInsertedRef.current = "";
        return;
      }
      if (!chunk) return;

      const el = textareaRef.current;
      const normalized = chunk.trim().toLowerCase().replace(/\s+/g, " ");
      const now = Date.now();

      if (
        normalized &&
        lastChunkRef.current &&
        lastChunkRef.current.text === normalized &&
        now - lastChunkRef.current.at < 5000
      ) return;

      const lastNorm = lastInsertedRef.current.trim().toLowerCase().replace(/\s+/g, " ");
      const shouldReplace =
        !!lastNorm &&
        normalized.length > lastNorm.length &&
        normalized.startsWith(lastNorm) &&
        now - (lastChunkRef.current?.at ?? 0) < 6000;

      setInputValue((prev) => {
        const start = el?.selectionStart ?? prev.length;
        const end = el?.selectionEnd ?? prev.length;
        const effectiveStart = shouldReplace
          ? Math.max(0, start - lastInsertedRef.current.length)
          : start;
        const before = prev.slice(0, effectiveStart);
        const after = prev.slice(end);
        const sep = before.length > 0 && !before.endsWith(" ") ? " " : "";
        const inserted = sep + chunk;
        lastInsertedRef.current = inserted;
        lastChunkRef.current = { text: normalized, at: now };
        const next = before + inserted + after;
        requestAnimationFrame(() => {
          el?.setSelectionRange(before.length + inserted.length, before.length + inserted.length);
          el?.focus();
        });
        return next;
      });
    },
    [],
  );

  const { isListening, toggle: toggleMic, stop: stopMic } = useSpeechToText({
    onTranscript: applyTranscript,
    onPermissionDenied: () => {},
    onNoSpeech: () => {},
    onError: () => {},
  });

  useEffect(() => {
    if (isStreaming && isListening) stopMic();
  }, [isStreaming, isListening, stopMic]);

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
                ? { ...m, content: streamingTextRef.current, timestamp: Date.now() }
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
        )) {
          if (event === "token") {
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
            const finalMessages = nextMessages.map((m) =>
              m.id === assistantId
                ? { ...m, content: finalText, timestamp: Date.now() }
                : m,
            );
            setMessages(finalMessages);
            saveCurrentSession(finalMessages);
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name !== "AbortError") {
          setErrorText(err.message || "An unexpected error occurred");
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
    [saveCurrentSession, selectedModel],
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
      className="@container flex h-full flex-col bg-background"
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
                  No documents — answers from model knowledge only
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((message) => {
          if (!message.content && message.role === "assistant" && !isStreaming) return null;
          const isUser = message.role === "user";
          const isPlaceholder = message.role === "assistant" && !message.content && isStreaming;

          if (isUser) {
            return (
              <div
                key={message.id}
                className="fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-4 duration-150 [&>*]:col-start-2"
                data-role="user"
              >
                <div className="relative col-start-2 min-w-0">
                  <div
                    className="wrap-break-word px-4 py-3 text-white text-sm"
                    style={{ background: "#2a2a2a", borderRadius: "18px 18px 4px 18px" }}
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
              <div className="wrap-break-word px-2 text-foreground text-base leading-[1.65]">
                {isPlaceholder ? (
                  <span className="text-muted-foreground animate-pulse">Thinking…</span>
                ) : (
                  <ChatMarkdown content={message.content} citations={[]} />
                )}
              </div>

              {!isPlaceholder && message.content && (
                <div className="mt-1 ml-2 flex">
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
        <div className="sticky bottom-0 mx-auto mt-auto flex w-full max-w-(--thread-max-width) flex-col gap-3 overflow-visible bg-background pb-4 pt-3 md:pb-6">
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
            <div className="flex w-full items-center gap-2 rounded-full ring-1 ring-[#2e2e2e] ring-inset bg-[#1a1a1a] px-4 py-1 outline-none transition-shadow focus-within:ring-[#444444]">
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
                className="max-h-32 flex-1 resize-none bg-transparent py-2 text-sm text-white outline-none placeholder:text-[#555555] focus-visible:ring-0 disabled:opacity-50 overflow-y-hidden"
              />

              {/* Model selector */}
              <SelectPrimitive.Root value={selectedModel} onValueChange={setSelectedModel}>
                <SelectPrimitive.Trigger className="flex items-center gap-1 h-8 px-2 text-xs font-medium text-muted-foreground bg-transparent hover:bg-accent hover:text-accent-foreground rounded-md transition-colors outline-none whitespace-nowrap shrink-0 border-0">
                  <SelectPrimitive.Value />
                  <SelectPrimitive.Icon asChild>
                    <ChevronDownIcon className="h-3 w-3 opacity-50" />
                  </SelectPrimitive.Icon>
                </SelectPrimitive.Trigger>
                <SelectPrimitive.Portal>
                  <SelectPrimitive.Content
                    className="z-50 min-w-[180px] overflow-hidden rounded-md border border-[#2e2e2e] bg-[#1a1a1a] text-foreground shadow-md"
                    position="popper"
                    sideOffset={8}
                    align="end"
                  >
                    <SelectPrimitive.Viewport className="p-1">
                      {MODELS.map((m) => (
                        <SelectPrimitive.Item
                          key={m.id}
                          value={m.id}
                          textValue={m.name}
                          className="relative flex w-full cursor-default select-none items-center gap-2 rounded-lg py-2 pr-9 pl-3 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
                        >
                          <span className="absolute right-3 flex size-4 items-center justify-center">
                            <SelectPrimitive.ItemIndicator>
                              <CheckIcon className="size-4" />
                            </SelectPrimitive.ItemIndicator>
                          </span>
                          <SelectPrimitive.ItemText>
                            <span className="font-medium">{m.name}</span>
                          </SelectPrimitive.ItemText>
                          <span className="text-xs text-muted-foreground">{m.description}</span>
                        </SelectPrimitive.Item>
                      ))}
                    </SelectPrimitive.Viewport>
                  </SelectPrimitive.Content>
                </SelectPrimitive.Portal>
              </SelectPrimitive.Root>

              {/* White circle button: mic / send / stop */}
              <div className="relative shrink-0 size-9 rounded-full bg-white">
                {/* Mic — shown when empty and not streaming */}
                <button
                  type="button"
                  aria-label={isListening ? "Stop listening" : "Voice input"}
                  onClick={() => { if (!isStreaming) toggleMic(); }}
                  className={cn(
                    "absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150",
                    (!isEmpty || isStreaming) ? "scale-0 opacity-0 pointer-events-none" : "",
                    isListening ? "bg-red-500" : "",
                  )}
                >
                  <MicIcon className={cn("size-4 text-black", isListening ? "animate-pulse" : "")} />
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
