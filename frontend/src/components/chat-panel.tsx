"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, LockBusyError } from "@/lib/api-fetch";
import { useAppDispatch, useAppState } from "@/context/app-context";

/** Last status line before the assistant message; everything before this is status. */
const MESSAGE_START_STATUS = "Generating answer...";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  parts: Array<{ type: "text"; text: string }>;
};

function getMessageText(message: ChatMessage): string {
  return message.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

function generateId(): string {
  return Math.random().toString(36).slice(2, 12);
}

/**
 * Chat panel with custom stream parsing: status lines go to the status
 * indicator, only the assistant reply is shown in the bubble, and the
 * reply is streamed token-by-token.
 */
export function ChatPanel() {
  const dispatch = useAppDispatch();
  const { statusMessage, errorMessage, isLockBusy } = useAppState();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Sync status when not streaming (e.g. clear when idle)
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

  const sendMessage = useCallback(
    async (text: string) => {
      dispatch({ type: "CLEAR_ERROR" });
      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        parts: [{ type: "text", text }],
      };
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: "assistant",
        parts: [{ type: "text", text: "" }],
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
      };

      // Use direct backend URL when set (bypasses Next.js proxy for proper streaming)
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
            if (j?.error?.message) dispatch({ type: "SET_ERROR", message: j.error.message });
            else dispatch({ type: "SET_ERROR", message: msg });
          } catch {
            dispatch({ type: "SET_ERROR", message: msg });
          }
          setIsStreaming(false);
          setMessages((prev) => prev.slice(0, -1));
          dispatch({ type: "CHAT_FINISH" });
          return;
        }

        // Parse stream: status lines → status indicator, rest → assistant message.
        // Next.js rewrites can buffer the response, so we support both streaming (body) and full text fallback.
        const processBuffer = (
          buffer: string,
          messageMode: { current: boolean },
          flushMessageContent: (chunk: string) => void
        ) => {
          let newlineIdx: number;
          while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, newlineIdx);
            buffer = buffer.slice(newlineIdx + 1);

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
                parts: [{ type: "text" as const, text: last.parts[0].text + chunk }],
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

            if (messageMode.current && buffer && !buffer.startsWith("Error:")) {
              flushChunk(buffer);
              buffer = "";
            }
          }
        } else {
          // Fallback when body is null (e.g. some proxies buffer the response)
          const fullText = await response.text();
          buffer = fullText;
          const result = processBuffer(buffer, messageMode, flushChunk);
          buffer = result.buffer;
          if (!result.done && messageMode.current && buffer && !buffer.startsWith("Error:")) {
            flushChunk(buffer);
          }
        }

        // Flush any remaining buffer as message content
        if (messageMode.current && buffer.trim() && !buffer.startsWith("Error:")) {
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
        abortRef.current = null;
        setIsStreaming(false);
        dispatch({ type: "CHAT_FINISH" });
      }
    },
    [messages, dispatch]
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
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
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

        {messages.map((message) => {
          const text = getMessageText(message);
          const isAssistantPlaceholder =
            message.role === "assistant" && !text && isStreaming;
          if (!text && message.role === "assistant" && !isStreaming) return null;
          return (
            <div
              key={message.id}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-xl px-4 py-3 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-100 border border-gray-700"
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed min-h-[1.5em]">
                  {isAssistantPlaceholder ? (
                    <span className="text-gray-500 animate-pulse">...</span>
                  ) : (
                    text
                  )}
                </div>
              </div>
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

      <div className="border-t border-gray-800 px-4 py-3">
        <form onSubmit={onSubmit} className="flex gap-2">
          <input
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              isActive
                ? "Waiting for response..."
                : "Ask a question about your documents..."
            }
            disabled={isActive}
            className="flex-1 rounded-lg bg-gray-800 border border-gray-700 px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {isActive ? (
            <button
              type="button"
              onClick={stop}
              className="px-4 py-2.5 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium text-white transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={isSubmitDisabled}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg text-sm font-medium text-white transition-colors"
            >
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
