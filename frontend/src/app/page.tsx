// ─── Two-panel chat layout ───────────────────────────────────────────────────
// SourcePanel (left) + assistant-ui Thread (right).
// CitationViewerModal overlays when activeCitation is set in AppContext.
//
// Layout notes:
//   Thread requires h-dvh on the parent and h-full on its direct container
//   so that the input bar anchors to the bottom via the Thread's internal
//   flex layout.  Parent must be overflow-hidden to prevent scroll bleed.
//
// Busy/running state is read via useThread((t) => t.isRunning) from
// @assistant-ui/react — AppContext does NOT track busy state.
//
// Endpoint: /api/chat is proxied by next.config.ts → http://127.0.0.1:8000/api/chat
// ──────────────────────────────────────────────────────────────────────────────

"use client";

import { useCallback, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { SourcePanel } from "@/components/source-panel";
import { useAppDispatch, useAppState } from "@/context/app-context";
import { parseCustomEvent } from "@/lib/event-parser";

/**
 * Full two-panel layout:
 *
 * ┌──────────────────────┬────────────────────────────────┐
 * │  SourcePanel (30%)   │  assistant-ui Thread (70%)     │
 * │  - source list       │  - streaming via /api/chat     │
 * │  - ingest modal      │  - ChatMarkdown + citations    │
 * └──────────────────────┴────────────────────────────────┘
 *
 * CitationViewerModal overlays the whole layout (absolute inset-0) when
 * activeCitation is non-null in AppContext.
 */
export default function Page() {
  const dispatch = useAppDispatch();
  const { activeCitation } = useAppState();

  // Source panel collapse state
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // Source IDs selected in the panel (passed down for UI checkbox state)
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);

  /**
   * Tracks whether we have already dispatched QUERY_STARTED for the current
   * streaming session.  Reset to false on every stream finish so the next
   * query triggers another QUERY_STARTED (which clears per-query state).
   */
  const streamStartedRef = useRef(false);

  /**
   * onData — fires for every custom data part emitted by the backend.
   *
   * AI SDK v6 "onData" pattern (confirmed via Context7):
   *   dataPart.type  → e.g. "data-status", "data-intent", "data-sources", …
   *   dataPart.data  → inner payload object from the backend annotation
   *
   * On the first call for a new query we dispatch QUERY_STARTED which
   * clears per-query state (lastIntent, lastSources, citations, statusMessage)
   * from the previous query before new stream events arrive.
   * Busy/running state is NOT tracked in AppContext — use
   * useThread((t) => t.isRunning) from @assistant-ui/react.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleData = useCallback((dataPart: any) => {
    // Detect query start: first data part of a new stream
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
          event.status !== "Generating answer..."
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
        const intentLabel = event.intent.charAt(0).toUpperCase() + event.intent.slice(1);
        const pct = Math.round(event.confidence * 100);
        dispatch({ type: "ADD_THINKING_STEP", message: `Intent identified: ${intentLabel} (${pct}% confidence)` });
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
      // "finish-step" and other internal protocol events are ignored here
    }
  }, [dispatch]);

  /**
   * onFinish — fires once when the entire stream completes.
   * Clears the statusMessage and resets the stream-started guard for the
   * next query.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleFinish = useCallback((_message: any) => {
    streamStartedRef.current = false;
    dispatch({ type: "QUERY_FINISHED" });
  }, [dispatch]);

  // useChatRuntime points at /api/chat which next.config.ts rewrites to
  // http://127.0.0.1:8000/api/chat — no CORS issues, no extra proxy route needed.
  // Passes all standard useChat options through to the underlying useChat hook.
  //
  // body is passed to AssistantChatTransport (HttpChatTransportInitOptions.body)
  // and merged into every request.  The backend ChatRequest.data field reads it.
  //   - citations_enabled: always true so the backend injects citation prompt rules
  //   - source_ids: the user-selected source filter (empty array = all sources)
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
      body: {
        data: {
          citations_enabled: true,
          source_ids: selectedSourceIds,
        },
      },
    }),
    onData: handleData,
    onFinish: handleFinish,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {/*
        h-dvh: full dynamic viewport height (handles iOS browser chrome correctly).
        overflow-hidden: prevents scroll bleed from child columns.
        relative: establishes stacking context for the CitationViewerModal overlay.
      */}
      <div className="flex h-dvh bg-background text-foreground overflow-hidden relative">

        {/* Left: Source Panel — 30% width, independently scrollable */}
        {!isPanelCollapsed ? (
          <aside className="w-[30%] min-w-60 max-w-sm border-r shrink-0 overflow-hidden flex flex-col" style={{ borderRightColor: "#1e1e1e" }}>
            <SourcePanel
              selectedSourceIds={selectedSourceIds}
              onSelectedSourceIdsChange={setSelectedSourceIds}
              onCollapse={() => setIsPanelCollapsed(true)}
            />
          </aside>
        ) : (
          /* Collapsed state: thin strip with expand button */
          <aside className="w-10 border-r shrink-0 flex flex-col items-center pt-3" style={{ borderRightColor: "#1e1e1e" }}>
            <button
              onClick={() => setIsPanelCollapsed(false)}
              className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors"
              title="Expand sources panel"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </aside>
        )}

        {/* Right: assistant-ui Thread — fills remaining width, h-full for input bar anchor */}
        <main className="flex-1 min-w-0 h-full">
          <Thread />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
}
