import { ChatMarkdownRendererWithSmooth } from "@/components/assistant-ui/chat-markdown-renderer";
import { ModelSelector } from "@/components/assistant-ui/model-selector";
import { IntentSelector } from "@/components/assistant-ui/intent-selector";
import { ThinkingPanel } from "@/components/assistant-ui/thinking-panel";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { MessageReferences } from "@/components/assistant-ui/message-references";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { NumberTicker } from "@/components/ui/number-ticker";
import { TypewriterText } from "@/components/ui/typewriter-text";
import { useAppState } from "@/context/app-context";
import { useIndexedStats } from "@/hooks/useIndexedStats";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { useSystemRam } from "@/hooks/useSystemRam";
import { cn } from "@/lib/utils";
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  SuggestionPrimitive,
  ThreadPrimitive,
  useAssistantApi,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  BrainIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  MicIcon,
  MoreHorizontalIcon,
  PencilIcon,
  RefreshCwIcon,
  SquareIcon,
  ThumbsUpIcon,
  ThumbsDownIcon,
} from "lucide-react";
import { type FC, useCallback, useEffect, useMemo, useRef, useState } from "react";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container flex h-full flex-col"
      style={{
        ["--thread-max-width" as string]: "45rem",
      }}
    >
      <ThreadPrimitive.Viewport
        turnAnchor="top"
        className="aui-thread-viewport relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll scroll-smooth px-4 pt-4"
      >
        <AuiIf condition={(s) => s.thread.isEmpty}>
          <ThreadWelcome />
        </AuiIf>

        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            EditComposer,
            AssistantMessage,
          }}
        />

        <ThreadPrimitive.ViewportFooter className="aui-thread-viewport-footer sticky bottom-0 mx-auto mt-auto flex w-full max-w-(--thread-max-width) flex-col gap-3 overflow-visible bg-transparent pb-4 pt-3 md:pb-6">
          <ThreadScrollToBottom />
          <Composer />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="aui-thread-scroll-to-bottom absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#2e2e2e] text-white"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  const { chatMode } = useAppState();
  const { sourceCount, estimatedTokens } = useIndexedStats();

  const hasData = sourceCount > 0;
  // Round to nearest 1k for a clean read; floor at 1k so it never shows "0 tokens"
  const displayTokens = estimatedTokens > 0
    ? Math.max(1000, Math.round(estimatedTokens / 1000) * 1000)
    : 0;

  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-center">
        <div className="aui-thread-welcome-message flex size-full flex-col justify-center px-4">
          <TypewriterText
            key={chatMode}
            text="Welcome to Corpus!"
            typingSpeed={80}
            className="aui-thread-welcome-message-inner font-semibold text-2xl"
          />
          {hasData ? (
            <p className="aui-thread-welcome-message-inner text-muted-foreground text-xl">
              <NumberTicker
                value={sourceCount}
                className="text-muted-foreground"
              />
              {" "}
              {sourceCount === 1 ? "source" : "sources"}
              {" · "}
              <NumberTicker
                value={displayTokens}
                className="text-muted-foreground"
              />
              {" tokens indexed"}
            </p>
          ) : (
            <p className="aui-thread-welcome-message-inner text-muted-foreground text-xl">
              Ingest documents to get started
            </p>
          )}
        </div>
      </div>
      <ThreadSuggestions />
    </div>
  );
};

const ThreadSuggestions: FC = () => {
  return (
    <div className="aui-thread-welcome-suggestions grid w-full @md:grid-cols-2 gap-2 pb-4">
      <ThreadPrimitive.Suggestions
        components={{
          Suggestion: ThreadSuggestionItem,
        }}
      />
    </div>
  );
};

const ThreadSuggestionItem: FC = () => {
  return (
    <div className="aui-thread-welcome-suggestion-display @md:nth-[n+3]:block nth-[n+3]:hidden">
      <SuggestionPrimitive.Trigger send asChild>
        <Button
          variant="ghost"
          className="aui-thread-welcome-suggestion h-auto w-full @md:flex-col flex-wrap items-start justify-start gap-1 rounded-2xl border border-[#2a2a2a] hover:border-[#3a3a3a] bg-[#1a1a1a] hover:bg-[#222222] px-4 py-3 text-left text-sm transition-colors"
        >
          <span className="aui-thread-welcome-suggestion-text-1 font-medium">
            <SuggestionPrimitive.Title />
          </span>
          <span className="aui-thread-welcome-suggestion-text-2 text-muted-foreground">
            <SuggestionPrimitive.Description />
          </span>
        </Button>
      </SuggestionPrimitive.Trigger>
    </div>
  );
};


const ALL_MODES = [
  { id: "regular", name: "Regular", description: "Qwen3.5-35B-A3B" },
  { id: "deep-research", name: "Deep Research", description: "Qwen3.5-35B-A3B (Deep Retrieval + Thinking)", minRamGb: 48 },
];

const ComposerSpeechButton: FC<{ disabled?: boolean }> = ({ disabled = false }) => {
  const aui = useAui();
  const composerText = useAuiState((s) => s.composer.text);
  const [errorText, setErrorText] = useState<string>("");
  const clearErrorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastChunkRef = useRef<{ text: string; at: number } | null>(null);
  const lastInsertedRef = useRef<string>("");

  const showError = useCallback((message: string) => {
    setErrorText(message);
    if (clearErrorTimerRef.current) clearTimeout(clearErrorTimerRef.current);
    clearErrorTimerRef.current = setTimeout(() => setErrorText(""), 3500);
  }, []);

  useEffect(() => {
    return () => {
      if (clearErrorTimerRef.current) clearTimeout(clearErrorTimerRef.current);
    };
  }, []);

  const applyTranscript = useCallback((chunk: string, isFinal: boolean) => {
    if (isFinal && !chunk) {
      lastChunkRef.current = null;
      lastInsertedRef.current = "";
      return;
    }
    const inputEl = document.querySelector("textarea.aui-composer-input, input.aui-composer-input") as HTMLTextAreaElement | HTMLInputElement | null;
    if (!chunk) return;

    const normalized = chunk.trim().toLowerCase().replace(/\s+/g, " ");
    const now = Date.now();
    if (
      normalized &&
      lastChunkRef.current &&
      lastChunkRef.current.text === normalized &&
      now - lastChunkRef.current.at < 5000
    ) {
      return;
    }

    const lastNorm = lastInsertedRef.current.trim().toLowerCase().replace(/\s+/g, " ");
    const shouldReplacePrevious =
      !!lastNorm &&
      normalized.length > lastNorm.length &&
      normalized.startsWith(lastNorm) &&
      now - (lastChunkRef.current?.at ?? 0) < 6000;

    const current = composerText ?? "";
    const start = inputEl?.selectionStart ?? current.length;
    const end = inputEl?.selectionEnd ?? current.length;
    const effectiveStart = shouldReplacePrevious
      ? Math.max(0, start - lastInsertedRef.current.length)
      : start;
    const before = current.slice(0, effectiveStart);
    const after = current.slice(end);
    const sep = before.length > 0 && !before.endsWith(" ") ? " " : "";
    const inserted = sep + chunk;
    const nextValue = before + inserted + after;

    aui.composer().setText(nextValue);
    const nextPos = before.length + inserted.length;
    requestAnimationFrame(() => {
      inputEl?.setSelectionRange(nextPos, nextPos);
      inputEl?.focus();
    });

    lastInsertedRef.current = inserted;
    lastChunkRef.current = { text: normalized, at: now };
  }, [aui, composerText]);

  const { isListening, toggle, stop } = useSpeechToText({
    onTranscript: applyTranscript,
    onPermissionDenied: () => showError("Mic access denied. Allow microphone access."),
    onNoSpeech: () => showError("No speech heard. Try again."),
    onError: (msg) => showError(msg),
  });

  useEffect(() => {
    if (disabled && isListening) {
      stop();
    }
  }, [disabled, isListening, stop]);

  const onClick = useCallback(() => {
    if (disabled) {
      showError("Wait for the current response to finish.");
      return;
    }
    setErrorText("");
    toggle();
  }, [disabled, showError, toggle]);

  return (
    <>
      <button
        type="button"
        aria-label={isListening ? "Stop listening" : "Voice input"}
        onClick={onClick}
        className={cn(
          "absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150 group-data-[empty=false]/composer:scale-0 group-data-[empty=false]/composer:opacity-0 group-data-[empty=false]/composer:pointer-events-none group-data-[running=true]/composer:scale-0 group-data-[running=true]/composer:opacity-0 group-data-[running=true]/composer:pointer-events-none",
          isListening ? "bg-red-500" : "",
        )}
      >
        <MicIcon className={cn("size-4 text-black", isListening ? "animate-pulse" : "")} />
      </button>
      {errorText ? (
        <div className="absolute -top-10 right-0 z-30 rounded-md border border-red-700 bg-red-900 px-2 py-1 text-[11px] text-red-100 whitespace-nowrap">
          {errorText}
        </div>
      ) : null}
    </>
  );
};

const Composer: FC = () => {
  const aui = useAui();
  const api = useAssistantApi();
  const isEmpty = useAuiState((s) => s.composer.isEmpty);
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const systemRamGb = useSystemRam();
  const canThink = systemRamGb !== null && systemRamGb >= 48;
  const [thinkingEnabled, setThinkingEnabled] = useState<boolean | null>(null);

  // Gate modes by available RAM
  const modes = useMemo(
    () =>
      ALL_MODES.filter(
        (m) => !m.minRamGb || (systemRamGb !== null && systemRamGb >= m.minRamGb),
      ),
    [systemRamGb],
  );

  // Register enableThinking in model context so the backend receives it
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const config = { config: { enableThinking: thinkingEnabled } as any };
    return api.modelContext().register({ getModelContext: () => config });
  }, [api, thinkingEnabled]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    // Reset scrollTop so the placeholder/text is never clipped on initial render
    el.scrollTop = 0;
    // Also trigger a synthetic input event so any auto-resize logic recalculates height
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, []);

  return (
    <ComposerPrimitive.Root
      className="aui-composer-root group/composer relative flex w-full flex-col"
      data-empty={isEmpty}
      data-running={isRunning}
    >
      {/* Input pill */}
      <div
        className="aui-composer-attachment-dropzone flex w-full items-end gap-2 rounded-3xl px-4 py-1 outline-none transition-shadow"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.12)",
          boxShadow: "0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)",
          backdropFilter: "blur(12px)",
        }}
      >
        {/* Text input — grows to fill space */}
        <ComposerPrimitive.Input
          ref={inputRef}
          placeholder="What do you want to know?"
          className="aui-composer-input max-h-40 flex-1 resize-none overflow-y-auto bg-transparent py-2 text-sm text-white outline-none placeholder:text-[#555555] focus-visible:ring-0"
          rows={1}
          autoFocus
          aria-label="Message input"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!isRunning) {
                aui.composer().send();
              }
            }
          }}
        />

        {/* Think toggle — visible when system has 48GB+ RAM */}
        {canThink && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() =>
                  setThinkingEnabled((v) =>
                    v === null ? true : v === true ? false : null,
                  )
                }
                className={cn(
                  "shrink-0 inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs leading-none font-medium transition-colors hover:bg-white/10",
                  thinkingEnabled === true
                    ? "text-blue-400 ring-1 ring-inset ring-blue-400/40"
                    : thinkingEnabled === false
                      ? "text-muted-foreground/40"
                      : "text-muted-foreground",
                )}
              >
                <BrainIcon className="size-3.5 shrink-0" />
                <span className={cn(thinkingEnabled === false && "line-through")}>Think</span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">
              {thinkingEnabled === true
                ? "Thinking forced on — click to force off"
                : thinkingEnabled === false
                  ? "Thinking forced off — click for auto"
                  : "Thinking auto (intent-driven) — click to force on"}
            </TooltipContent>
          </Tooltip>
        )}

        {/* Model selector — always visible */}
        <ModelSelector
          models={modes}
          defaultValue="regular"
          variant="ghost"
          size="sm"
        />

        {/* Intent selector — always visible, respects Auto default */}
        <IntentSelector
          variant="ghost"
          size="sm"
        />

        {/* Animated 3-state button */}
        <div className="relative shrink-0 size-9 rounded-full bg-white">
          {/* Mic — visible only when empty & not running */}
          <ComposerSpeechButton disabled={isRunning} />

          {/* Send — visible when text present & not running */}
          <ComposerPrimitive.Send
            className="absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150 group-data-[empty=true]/composer:scale-0 group-data-[empty=true]/composer:opacity-0 group-data-[empty=true]/composer:pointer-events-none group-data-[running=true]/composer:scale-0 group-data-[running=true]/composer:opacity-0 group-data-[running=true]/composer:pointer-events-none"
            aria-label="Send message"
          >
            <ArrowUpIcon className="size-4 text-black" />
          </ComposerPrimitive.Send>

          {/* Cancel — visible only when running */}
          <ComposerPrimitive.Cancel
            className="absolute inset-0 flex items-center justify-center rounded-full transition-all duration-150 group-data-[running=false]/composer:scale-0 group-data-[running=false]/composer:opacity-0 group-data-[running=false]/composer:pointer-events-none"
            aria-label="Stop generating"
          >
            <SquareIcon className="size-3 fill-black text-black" />
          </ComposerPrimitive.Cancel>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const AssistantMessage: FC = () => {
  // compute the current accumulated text so we know when to show the bubble
  const messageText = useAuiState((s) =>
    s.message.parts.reduce((acc: string, part) => {
      if (part.type !== "text") return acc;
      if (!("text" in part) || typeof (part as { text?: unknown }).text !== "string") return acc;
      return acc + (part as { text: string }).text;
    }, ""),
  );

  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-4 duration-150"
      data-role="assistant"
    >
      <ThinkingPanel />

      {/* only render the bubble once text begins to arrive */}
      {messageText.length > 0 && (
        <div
          className="aui-assistant-message-content wrap-break-word px-4 py-3 text-foreground text-base leading-[1.65] bg-white/5 backdrop-blur-sm [&_p]:my-3 [&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-7 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:pl-7 [&_li]:leading-relaxed [&_li+li]:mt-2.5 [&_li>p]:my-0"
          style={{
            borderRadius: "18px 18px 18px 4px",
          }}
        >
          <MessagePrimitive.Parts
            components={{
              Text: ChatMarkdownRendererWithSmooth,
              tools: { Fallback: ToolFallback },
            }}
          />
          <MessageReferences />
          <MessageError />
        </div>
      )}

      <div className="aui-assistant-message-footer mt-1 ml-2 flex">
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  const isRunning = useAuiState((s) => s.message.status?.type === "running");
  const messageText = useAuiState((s) =>
    s.message.parts.reduce((acc: string, part) => {
      if (part.type !== "text") return acc;
      if (!("text" in part) || typeof (part as { text?: unknown }).text !== "string") return acc;
      return acc + (part as { text: string }).text;
    }, ""),
  );
  const messageId = useAuiState((s) => s.message.id);
  const { traceInfoByMessage } = useAppState();
  const traceInfo = traceInfoByMessage[messageId];
  const [isCleanlyCopied, setIsCleanlyCopied] = useState(false);
  const [feedbackState, setFeedbackState] = useState<"up" | "down" | null>(null);

  const handleCleanCopy = useCallback(() => {
    // Strip [N] citation markers and collapse any resulting double spaces
    const clean = messageText
      .replace(/\s*\[\d+\]\s*/g, " ")
      .replace(/ {2,}/g, " ")
      .trim();
    navigator.clipboard.writeText(clean).then(() => {
      setIsCleanlyCopied(true);
      setTimeout(() => setIsCleanlyCopied(false), 2000);
    });
  }, [messageText]);

  const handleFeedback = useCallback((label: "👍" | "👎", score: number) => {
    const newState = (label === "👍" ? "up" : "down");
    setFeedbackState((prev) => (prev === newState ? null : newState));
    
    if (traceInfo) {
      fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trace_id: traceInfo.traceId,
          span_id: traceInfo.spanId,
          label,
          score,
        }),
      }).catch(console.error);
    }
  }, [traceInfo]);

  return (
    <ActionBarPrimitive.Root
      autohide="not-last"
      autohideFloat="single-branch"
      className="aui-assistant-action-bar-root col-start-3 row-start-2 -ml-1 flex gap-1 text-muted-foreground data-floating:absolute data-floating:rounded-md data-floating:border data-floating:bg-background data-floating:p-1 data-floating:shadow-sm"
    >
      {!isRunning && (
        <TooltipIconButton tooltip="Copy" side="top" onClick={handleCleanCopy}>
          {isCleanlyCopied ? <CheckIcon /> : <CopyIcon />}
        </TooltipIconButton>
      )}
      {!isRunning && (
        <ActionBarPrimitive.Reload asChild>
          <TooltipIconButton tooltip="Refresh" side="top">
            <RefreshCwIcon />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
      )}
      {!isRunning && (
        <TooltipIconButton
          tooltip="Good response"
          side="top"
          disabled={!traceInfo}
          onClick={() => handleFeedback("👍", 1.0)}
          className={feedbackState === "up" ? "text-green-400" : undefined}
        >
          <ThumbsUpIcon />
        </TooltipIconButton>
      )}
      {!isRunning && (
        <TooltipIconButton
          tooltip="Poor response"
          side="top"
          disabled={!traceInfo}
          onClick={() => handleFeedback("👎", 0.0)}
          className={feedbackState === "down" ? "text-red-400" : undefined}
        >
          <ThumbsDownIcon />
        </TooltipIconButton>
      )}
      <MessageTimingBadge />
      {!isRunning && (
        <ActionBarMorePrimitive.Root>
          <ActionBarMorePrimitive.Trigger asChild>
            <TooltipIconButton
              tooltip="More"
              side="top"
              className="data-[state=open]:bg-accent"
            >
              <MoreHorizontalIcon />
            </TooltipIconButton>
          </ActionBarMorePrimitive.Trigger>
          <ActionBarMorePrimitive.Content
            side="bottom"
            align="start"
            className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md border-transparent bg-popover p-1 text-popover-foreground shadow-md"
          >
            <ActionBarPrimitive.ExportMarkdown asChild>
              <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                <DownloadIcon className="size-4" />
                Export as Markdown
              </ActionBarMorePrimitive.Item>
            </ActionBarPrimitive.ExportMarkdown>
          </ActionBarMorePrimitive.Content>
        </ActionBarMorePrimitive.Root>
      )}
    </ActionBarPrimitive.Root>
  );
};

const MessageTimingBadge: FC = () => {
  const messageId = useAuiState((s) => s.message.id);
  const isLast = useAuiState((s) => s.message.isLast);
  const isRunning = useAuiState((s) => s.message.status?.type === "running");
  const text = useAuiState((s) =>
    s.message.parts.reduce((allText, part) => {
      if (part.type !== "text") return allText;
      if (!("text" in part) || typeof part.text !== "string") return allText;
      return allText + part.text;
    }, ""),
  );

  const [streamStartAt, setStreamStartAt] = useState<number | undefined>(undefined);
  const [firstTokenAt, setFirstTokenAt] = useState<number | undefined>(undefined);
  const [completedAt, setCompletedAt] = useState<number | undefined>(undefined);

  useEffect(() => {
    queueMicrotask(() => {
      setStreamStartAt(undefined);
      setFirstTokenAt(undefined);
      setCompletedAt(undefined);
    });
  }, [messageId]);

  useEffect(() => {
    if (!isRunning || streamStartAt !== undefined) return;
    queueMicrotask(() => setStreamStartAt(Date.now()));
  }, [isRunning, streamStartAt]);

  useEffect(() => {
    if (!isRunning || streamStartAt === undefined || firstTokenAt !== undefined || text.length === 0) return;
    queueMicrotask(() => setFirstTokenAt(Date.now()));
  }, [firstTokenAt, isRunning, streamStartAt, text]);

  useEffect(() => {
    if (isRunning || streamStartAt === undefined || completedAt !== undefined) return;
    queueMicrotask(() => setCompletedAt(Date.now()));
  }, [completedAt, isRunning, streamStartAt]);

  if (!isLast) return null;
  if (isRunning) return null;
  if (streamStartAt === undefined || completedAt === undefined) return null;

  const totalMs = Math.max(0, completedAt - streamStartAt);

  const estimatedTokenCount = Math.max(1, Math.round(text.length / 4));
  const tokensPerSecond =
    totalMs >= 400 && estimatedTokenCount >= 8
      ? estimatedTokenCount / (totalMs / 1000)
      : undefined;
  const retrievalMs =
    streamStartAt !== undefined && firstTokenAt !== undefined
      ? Math.max(0, firstTokenAt - streamStartAt)
      : undefined;

  const formatDetailedMs = (ms: number) =>
    ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`;
  const formatBadgeMs = (ms: number) => `${(ms / 1000).toFixed(1)}s`;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex h-6 items-center justify-center whitespace-nowrap rounded-md px-2 text-sm font-medium text-muted-foreground ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground"
          aria-label="Message timing"
        >
          <span className="font-mono leading-none">{formatBadgeMs(totalMs)}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        align="center"
        sideOffset={8}
        className="min-w-52 border-[#2e2e2e] bg-[#1a1a1a] text-white shadow-xl"
      >
        <div className="space-y-1.5 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">Total</span>
            <span className="font-mono text-foreground">{formatDetailedMs(totalMs)}</span>
          </div>
          {tokensPerSecond !== undefined && Number.isFinite(tokensPerSecond) && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Speed</span>
              <span className="font-mono text-foreground">{tokensPerSecond.toFixed(1)} tok/s</span>
            </div>
          )}
          {retrievalMs !== undefined && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Retrieval</span>
              <span className="font-mono text-foreground">{formatDetailedMs(retrievalMs)}</span>
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-4 duration-150 [&:where(>*)]:col-start-2"
      data-role="user"
    >
      <div className="aui-user-message-content-wrapper relative col-start-2 min-w-0">
        <div
          className="aui-user-message-content wrap-break-word px-4 py-3 text-white text-base leading-[1.65] bg-white/10 backdrop-blur-lg"
          style={{
            borderRadius: "18px 18px 4px 18px",
          }}
        >
          <MessagePrimitive.Parts />
        </div>
        <div className="aui-user-action-bar-wrapper absolute top-1/2 left-0 -translate-x-full -translate-y-1/2 pr-2">
          <UserActionBar />
        </div>
      </div>

      <BranchPicker className="aui-user-branch-picker col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="aui-user-action-bar-root flex flex-col items-end"
    >
      <ActionBarPrimitive.Edit asChild>
        <TooltipIconButton tooltip="Edit" className="aui-user-action-edit p-4">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-max-width) flex-col px-2 py-3">
      <ComposerPrimitive.Root
        className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.12)",
          boxShadow: "0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)",
          backdropFilter: "blur(12px)",
        }}
      >
        <ComposerPrimitive.Input
          className="aui-edit-composer-input min-h-14 w-full resize-none bg-transparent p-4 text-white text-sm outline-none placeholder:text-[#555555]"
          autoFocus
        />
        <div className="aui-edit-composer-footer mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel asChild>
            <Button variant="ghost" size="sm">
              Cancel
            </Button>
          </ComposerPrimitive.Cancel>
          <ComposerPrimitive.Send asChild>
            <Button size="sm">Update</Button>
          </ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "aui-branch-picker-root mr-2 -ml-2 inline-flex items-center text-muted-foreground text-xs",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
