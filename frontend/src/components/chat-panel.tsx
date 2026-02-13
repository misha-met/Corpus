"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, LockBusyError } from "@/lib/api-fetch";
import { useAppDispatch, useAppState } from "@/context/app-context";
import type { CitationEntry, CitationPayload } from "@/lib/api-client";

/** Last status line before the assistant message; everything before this is status. */
const MESSAGE_START_STATUS = "Generating answer...";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  parts: Array<{ type: "text"; text: string }>;
  timestamp: number;
  /** Source IDs referenced in this message (populated from stream annotations) */
  sourceIds?: string[];
  /** Structured citation list parsed from the CITATIONS: stream line */
  citations?: CitationEntry[];
};

/** Render text with citation numbers as superscript.
 *  Uses the original [N] number as the display label (not a sequential counter)
 *  so buttons match the Sources header.  Unresolved [N] (N > citations.length)
 *  is rendered as plain text.  Duplicate [N] occurrences get the same label. */
function renderTextWithCitations(
  text: string,
  citations: CitationEntry[] | undefined,
  onCitationClick: (payload: CitationPayload) => void
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const regex = /\[([^\]]+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Text before citation
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>
      );
    }

    const raw = match[1].trim();
    let payload: CitationPayload | null = null;
    let displayLabel = raw;

    // Resolve numbered [N]
    const numMatch = raw.match(/^(\d+)$/);
    if (numMatch && citations) {
      const idx = parseInt(numMatch[1], 10) - 1;
      if (idx >= 0 && idx < citations.length) {
        const c = citations[idx];
        displayLabel = numMatch[1];
        payload = {
          source_id: c.source_id,
          chunk_id: c.chunk_id,
          page_number: c.page_number,
          display_page: c.display_page,
          header_path: c.header_path,
          chunk_text: c.chunk_text,
        };
      }
    } else {
      // Try [SourceID, p. X]
      const pageMatch = raw.match(/^(.+?),\s*p\.\s*(\d+)$/);
      const sourceId = pageMatch ? pageMatch[1].trim() : raw;
      const entry = citations?.find((c) => c.source_id === sourceId);
      if (entry) {
        payload = {
          source_id: entry.source_id,
          chunk_id: entry.chunk_id,
          page_number: entry.page_number,
          display_page: entry.display_page,
          header_path: entry.header_path,
          chunk_text: entry.chunk_text,
        };
      }
    }

    if (payload) {
      parts.push(
        <button
          key={`cite-${match.index}`}
          onClick={(e) => {
            e.stopPropagation();
            onCitationClick(payload);
          }}
          className="inline-flex items-center justify-center w-4 h-4 text-[10px] font-bold text-blue-400 hover:text-blue-300 bg-blue-500/20 hover:bg-blue-500/30 rounded-full align-super ml-0.5 mr-0.5 cursor-pointer transition-colors"
          title={`View source: ${payload.source_id}`}
        >
          {displayLabel}
        </button>
      );
    } else {
      // Unresolved citation — render as plain text (no clickable button)
      parts.push(
        <span key={`text-${match.index}`}>{match[0]}</span>
      );
    }
    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex)}</span>);
  }

  return parts.length > 0 ? parts : [<span key="full">{text}</span>];
}

function getMessageText(message: ChatMessage): string {
  return message.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

function generateId(): string {
  return Math.random().toString(36).slice(2, 12);
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const isToday =
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear();
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  return isToday ? `Today \u2022 ${time}` : `${d.toLocaleDateString()} \u2022 ${time}`;
}

interface ChatPanelProps {
  selectedSourceIds: string[];
  sourceCount: number;
  onCitationClick: (payload: CitationPayload) => void;
}

/**
 * Chat panel with custom stream parsing, citation rendering,
 * action buttons, timestamps, and source count in input.
 */
export function ChatPanel({
  selectedSourceIds,
  sourceCount,
  onCitationClick,
}: ChatPanelProps) {
  const dispatch = useAppDispatch();
  const { statusMessage, errorMessage, isLockBusy } = useAppState();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  /** Holds citations parsed from current stream, attached to assistant message on finish */
  const pendingCitationsRef = useRef<CitationEntry[] | null>(null);

  // Sync status when not streaming
  useEffect(() => {
    if (!isStreaming && statusMessage) {
      dispatch({ type: "SET_STATUS", status: "" });
    }
  }, [isStreaming, statusMessage, dispatch]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const isActive = isStreaming;
  const isSubmitDisabled = isActive || !inputValue.trim();

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  async function copyToClipboard(messageId: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(messageId);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      /* clipboard not available */
    }
  }

  const sendMessage = useCallback(
    async (text: string) => {
      dispatch({ type: "CLEAR_ERROR" });
      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        parts: [{ type: "text", text }],
        timestamp: Date.now(),
      };
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: "assistant",
        parts: [{ type: "text", text: "" }],
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);
      dispatch({ type: "CHAT_START" });
      dispatch({ type: "SET_STATUS", status: "Sending query..." });

      const controller = new AbortController();
      abortRef.current = controller;

      const body = {
        messages: [
          ...messages.map((m) => ({
            role: m.role,
            parts: m.parts,
          })),
          { role: "user" as const, parts: [{ type: "text" as const, text }] },
        ],
        data: {
          source_ids: selectedSourceIds.length > 0 ? selectedSourceIds : undefined,
          citations_enabled: true,
        },
      };

      const chatUrl =
        typeof process.env.NEXT_PUBLIC_BACKEND_URL === "string" &&
        process.env.NEXT_PUBLIC_BACKEND_URL
          ? `${process.env.NEXT_PUBLIC_BACKEND_URL.replace(/\/$/, "")}/api/chat`
          : "/api/chat";

      try {
        const response = await apiFetch(chatUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!response.ok) {
          const msg = response.statusText || "Request failed";
          try {
            const j = await response.json();
            if (j?.error?.message)
              dispatch({ type: "SET_ERROR", message: j.error.message });
            else dispatch({ type: "SET_ERROR", message: msg });
          } catch {
            dispatch({ type: "SET_ERROR", message: msg });
          }
          setIsStreaming(false);
          setMessages((prev) => prev.slice(0, -1));
          dispatch({ type: "CHAT_FINISH" });
          return;
        }

        const processBuffer = (
          buffer: string,
          messageMode: { current: boolean },
          flushMessageContent: (chunk: string) => void
        ) => {
          let newlineIdx: number;
          while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, newlineIdx);
            buffer = buffer.slice(newlineIdx + 1);

            // Parse CITATIONS: line (emitted before "Generating answer...")
            if (line.startsWith("CITATIONS:")) {
              try {
                const json = line.slice("CITATIONS:".length);
                pendingCitationsRef.current = JSON.parse(json) as CitationEntry[];
              } catch {
                /* ignore malformed citations line */
              }
              continue;
            }

            if (!messageMode.current) {
              if (line.trim()) dispatch({ type: "SET_STATUS", status: line });
              if (line === MESSAGE_START_STATUS) messageMode.current = true;
              continue;
            }
            if (line === "" || line === " ") continue;
            if (line.startsWith("Error: ")) {
              dispatch({ type: "SET_ERROR", message: line.slice(7).trim() });
              return { done: true as const, buffer: "" };
            }
            flushMessageContent(line + "\n");
          }
          return { done: false as const, buffer };
        };

        const flushChunk = (chunk: string) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.parts[0]) {
              next[next.length - 1] = {
                ...last,
                parts: [
                  { type: "text" as const, text: last.parts[0].text + chunk },
                ],
                timestamp: Date.now(),
              };
            }
            return next;
          });
        };

        const messageMode = { current: false };
        let buffer = "";

        if (response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          streamLoop: while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const result = processBuffer(buffer, messageMode, flushChunk);
            buffer = result.buffer;
            if (result.done) break streamLoop;

            if (
              messageMode.current &&
              buffer &&
              !buffer.startsWith("Error:")
            ) {
              flushChunk(buffer);
              buffer = "";
            }
          }
        } else {
          const fullText = await response.text();
          buffer = fullText;
          const result = processBuffer(buffer, messageMode, flushChunk);
          buffer = result.buffer;
          if (
            !result.done &&
            messageMode.current &&
            buffer &&
            !buffer.startsWith("Error:")
          ) {
            flushChunk(buffer);
          }
        }

        if (
          messageMode.current &&
          buffer.trim() &&
          !buffer.startsWith("Error:")
        ) {
          flushChunk(buffer);
        }
      } catch (err) {
        if (err instanceof LockBusyError) {
          dispatch({
            type: "SET_ERROR",
            message: err.message,
            isLockBusy: true,
          });
        } else if (err instanceof Error && err.name !== "AbortError") {
          dispatch({
            type: "SET_ERROR",
            message: err.message || "An unexpected error occurred",
          });
        }
        if ((err as Error)?.name === "AbortError") {
          setMessages((prev) => prev.slice(0, -1));
        }
      } finally {
        // Attach parsed citations to the last assistant message
        if (pendingCitationsRef.current) {
          const cits = pendingCitationsRef.current;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              next[next.length - 1] = { ...last, citations: cits };
            }
            return next;
          });
          pendingCitationsRef.current = null;
        }
        abortRef.current = null;
        setIsStreaming(false);
        dispatch({ type: "CHAT_FINISH" });
      }
    },
    [messages, dispatch, selectedSourceIds]
  );

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (isSubmitDisabled) return;
    const text = inputValue.trim();
    setInputValue("");
    sendMessage(text);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit(e);
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Chat header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800 shrink-0">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide">
          Chat
        </h2>
        <div className="flex items-center gap-1">
          {/* Grid view icon (placeholder) */}
          <button
            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors"
            title="Grid view"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
              />
            </svg>
          </button>
          {/* Menu icon (placeholder) */}
          <button
            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors"
            title="More options"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center space-y-2">
              <p className="text-lg font-medium">
                Ask a question about your documents
              </p>
              <p className="text-sm text-gray-600">
                Your messages will appear here
              </p>
            </div>
          </div>
        )}

        {messages.map((message, idx) => {
          const text = getMessageText(message);
          const isAssistantPlaceholder =
            message.role === "assistant" && !text && isStreaming;
          if (!text && message.role === "assistant" && !isStreaming) return null;

          const isUser = message.role === "user";
          const msgCitations = message.citations ?? [];

          // Show timestamp between user/assistant pairs or at certain intervals
          const showTimestamp =
            idx === messages.length - 1 ||
            (idx < messages.length - 1 &&
              messages[idx + 1].role === "user" &&
              message.role === "assistant");

          return (
            <div key={message.id} className="space-y-1">
              {/* Message bubble */}
              <div
                className={`flex ${isUser ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    isUser
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800/80 text-gray-100 border border-gray-700/50"
                  }`}
                >
                  {/* Citation strip for assistant */}
                  {!isUser && msgCitations.length > 0 && (
                    <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-gray-700/50">
                      <span className="text-xs text-gray-400">Sources:</span>
                      <div className="flex items-center gap-1 flex-wrap">
                        {msgCitations.map((c) => (
                          <button
                            key={c.index}
                            onClick={() => onCitationClick({
                              source_id: c.source_id,
                              chunk_id: c.chunk_id,
                              page_number: c.page_number,
                              display_page: c.display_page,
                              header_path: c.header_path,
                              chunk_text: c.chunk_text,
                            })}
                            className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 text-[10px] font-bold text-blue-400 hover:text-blue-300 bg-blue-500/20 hover:bg-blue-500/30 rounded-full cursor-pointer transition-colors"
                            title={`View: ${c.source_id}`}
                          >
                            {c.index}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Message text */}
                  <div className="whitespace-pre-wrap text-sm leading-relaxed min-h-[1.5em]">
                    {isAssistantPlaceholder ? (
                      <span className="text-gray-500 animate-pulse">...</span>
                    ) : isUser ? (
                      text
                    ) : (
                      renderTextWithCitations(text, message.citations, onCitationClick)
                    )}
                  </div>
                </div>
              </div>

              {/* Action buttons for assistant messages */}
              {!isUser && text && !isAssistantPlaceholder && (
                <div className="flex items-center gap-0.5 pl-1">
                  {/* Save to note */}
                  <button
                    className="flex items-center gap-1 px-2 py-1 text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800/50 rounded transition-colors"
                    title="Save to note"
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
                      />
                    </svg>
                    <span>Save</span>
                  </button>

                  {/* Copy */}
                  <button
                    onClick={() => copyToClipboard(message.id, text)}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800/50 rounded transition-colors"
                    title="Copy message"
                  >
                    {copiedId === message.id ? (
                      <svg
                        className="w-3.5 h-3.5 text-green-400"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="w-3.5 h-3.5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                    )}
                    <span>{copiedId === message.id ? "Copied" : "Copy"}</span>
                  </button>

                  {/* Thumbs up */}
                  <button
                    className="p-1 text-gray-500 hover:text-gray-300 hover:bg-gray-800/50 rounded transition-colors"
                    title="Good response"
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14zm-9 11H4a2 2 0 01-2-2v-7a2 2 0 012-2h1"
                      />
                    </svg>
                  </button>

                  {/* Thumbs down */}
                  <button
                    className="p-1 text-gray-500 hover:text-gray-300 hover:bg-gray-800/50 rounded transition-colors"
                    title="Bad response"
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10zm9-13h1a2 2 0 012 2v7a2 2 0 01-2 2h-1"
                      />
                    </svg>
                  </button>
                </div>
              )}

              {/* Timestamp */}
              {showTimestamp && (
                <div className="text-center">
                  <span className="text-[10px] text-gray-600">
                    {formatTimestamp(message.timestamp)}
                  </span>
                </div>
              )}
            </div>
          );
        })}

        {isActive && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              {statusMessage || "Thinking..."}
            </div>
          </div>
        )}

        {errorMessage && (
          <div className="flex justify-center">
            <div
              className={`px-4 py-2 rounded-lg text-sm ${
                isLockBusy
                  ? "bg-yellow-900/50 text-yellow-200 border border-yellow-800"
                  : "bg-red-900/50 text-red-200 border border-red-800"
              }`}
            >
              {isLockBusy && (
                <span className="font-medium mr-1">Server busy:</span>
              )}
              {errorMessage}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-800 px-5 py-3">
        <form onSubmit={onSubmit} className="relative flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                isActive ? "Waiting for response..." : "Start typing..."
              }
              disabled={isActive}
              className="w-full rounded-xl bg-gray-800 border border-gray-700 pl-4 pr-24 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            />
            {/* Source count badge inside input */}
            <div className="absolute right-12 top-1/2 -translate-y-1/2 flex items-center">
              <span className="text-xs text-gray-500 bg-gray-700/50 px-2 py-0.5 rounded-full">
                {sourceCount} source{sourceCount !== 1 ? "s" : ""}
              </span>
            </div>
          </div>

          {/* Circular send / stop button */}
          {isActive ? (
            <button
              type="button"
              onClick={stop}
              className="w-10 h-10 flex items-center justify-center bg-red-600 hover:bg-red-700 rounded-full text-white transition-colors shrink-0"
              title="Stop"
            >
              <svg
                className="w-4 h-4"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={isSubmitDisabled}
              className="w-10 h-10 flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-full text-white transition-colors shrink-0"
              title="Send"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 12h14M12 5l7 7-7 7"
                />
              </svg>
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
