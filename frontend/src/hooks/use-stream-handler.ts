"use client";

import { useRef, useCallback } from "react";
import { useAppDispatch } from "@/context/app-context";
import { parseCustomEvent } from "@/lib/event-parser";

type AppDispatch = ReturnType<typeof useAppDispatch>;

/** "compare_sources" → "Compare Sources" (with en-GB spelling). */
function formatIntentLabel(intent: string): string {
  return intent
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
    .replace("Summarize", "Summarise")
    .replace("summarize", "summarise");
}

/**
 * Encapsulates the onData / onFinish callbacks that bridge the AI-SDK
 * stream to AppContext dispatch actions.
 *
 * `handleData`  — fires for every custom data-part emitted by the backend.
 * `handleFinish` — fires once when the entire stream completes.
 */
export function useStreamHandler(dispatch: AppDispatch) {
  const streamStartedRef = useRef(false);

  const handleData = useCallback(
    (dataPart: unknown) => {
      if (!streamStartedRef.current) {
        streamStartedRef.current = true;
        dispatch({ type: "QUERY_STARTED" });
      }

      const event = parseCustomEvent(dataPart);
      if (!event) return;

      switch (event.type) {
        case "status":
          dispatch({ type: "SET_STATUS", status: event.status });
          if (
            event.status !== "Building prompt..." &&
            event.status !== "Generating answer..." &&
            !event.status.startsWith("Using intent:")
          ) {
            dispatch({ type: "ADD_THINKING_STEP", message: event.status });
          }
          break;

        case "intent": {
          dispatch({
            type: "SET_INTENT",
            intent: event.intent,
            confidence: event.confidence,
            method: event.method,
          });
          const label = formatIntentLabel(event.intent);
          const pct = Math.round(event.confidence * 100);
          const msg =
            event.method === "manual"
              ? `Intent: ${label}`
              : `Intent identified: ${label} (${pct}% confidence)`;
          dispatch({ type: "ADD_THINKING_STEP", message: msg });
          break;
        }

        case "sources":
          dispatch({ type: "SET_SOURCES", sourceIds: event.sourceIds });
          break;

        case "citations":
          dispatch({ type: "SET_CITATIONS", citations: event.citations });
          break;

        case "error":
          dispatch({ type: "SET_ERROR", message: event.error.message });
          break;

        case "trace-id":
          dispatch({
            type: "SET_TRACE_INFO",
            traceId: event.traceId,
            spanId: event.spanId,
          });
          break;
      }
    },
    [dispatch],
  );

  const handleFinish = useCallback((): void => {
    streamStartedRef.current = false;
    dispatch({ type: "QUERY_FINISHED" });
  }, [dispatch]);

  return { handleData, handleFinish } as const;
}
