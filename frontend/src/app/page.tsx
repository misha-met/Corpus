// ─── Two-panel chat layout ───────────────────────────────────────────────────
// SourcePanel (left) + dual-mode right panel (RAG Thread or Free Chat).
// CitationViewerModal overlays when activeCitation is set in AppContext.
//
// Dual-mode approach:
//   Both <Thread> and <FreeformChatPanel> are always mounted.
//   Visibility is toggled with Tailwind `hidden` so neither panel loses state
//   when the user flips between modes.
//
// ──────────────────────────────────────────────────────────────────────────────

"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { AssistantRuntimeProvider, useAuiState } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { SourcePanel } from "@/components/source-panel";
import { FreeformChatPanel } from "@/components/freeform-chat-panel";
import { HistoryPanel } from "@/components/history-panel";
import { useAppDispatch, useAppState } from "@/context/app-context";
import { parseCustomEvent } from "@/lib/event-parser";
import type { ChatSession } from "@/lib/session-store";
import type { FreeChatMessage } from "@/lib/session-store";
import { Plus } from "lucide-react";
import { BackgroundBeams } from "@/components/ui/beams";
import { Meteors } from "@/components/ui/meteors";
import { RainBackground } from "@/components/ui/rain";
import { MeshGradientBackground } from "@/components/ui/mesh-gradient";
import { BackgroundPaths } from "@/components/ui/paths";
import { StarfieldBackground } from "@/components/ui/starfield";
import { ParticleBackground } from "@/components/ui/particles";
import { Leva } from "leva";
import { useTheme, type BackgroundTheme } from "@/context/theme-context";

function MessageIdTracker() {
  const dispatch = useAppDispatch();
  const lastAssistantMessageId = useAuiState((s) => {
    const lastMsg = s.thread.messages[s.thread.messages.length - 1];
    return lastMsg?.role === "assistant" ? lastMsg.id : null;
  });

  useEffect(() => {
    if (lastAssistantMessageId) {
      dispatch({ type: "SET_CURRENT_MESSAGE_ID", messageId: lastAssistantMessageId });
    }
  }, [lastAssistantMessageId, dispatch]);

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// RagArea — owns the chat runtime + Thread so it can be re-keyed for new chat.
// ─────────────────────────────────────────────────────────────────────────────
interface RagAreaProps {
  selectedSourceIds: string[];
  intentOverride: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onData: (dataPart: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onFinish: (message: any) => void;
}

function RagArea({ selectedSourceIds, intentOverride, onData, onFinish }: RagAreaProps) {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
      body: {
        data: {
          citations_enabled: true,
          source_ids: selectedSourceIds,
          intent_override: intentOverride,
        },
      },
    }),
    onData,
    onFinish,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MessageIdTracker />
      <Thread />
    </AssistantRuntimeProvider>
  );
}

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
  const { intentOverride, chatMode } = useAppState();
  const { theme, setTheme } = useTheme();
  const [themeOpen, setThemeOpen] = useState(false);

  // Close theme picker on Escape
  useEffect(() => {
    if (!themeOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setThemeOpen(false);
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [themeOpen]);

  const THEMES: { id: BackgroundTheme; label: string }[] = [
    { id: "none",      label: "Default" },
    { id: "beams",     label: "Beams" },
    { id: "meteors",   label: "Meteors" },
    { id: "rain",      label: "Rain" },
    { id: "mesh",      label: "Gradient" },
    { id: "paths",     label: "Paths" },
    { id: "starfield", label: "Starfield" },
    { id: "particles", label: "Particles" },
  ];

  // Source panel collapse state
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // Source IDs selected in the panel (passed down for UI checkbox state)
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);

  // History panel
  const [showHistory, setShowHistory] = useState(false);

  // New chat reset keys
  const [ragKey, setRagKey] = useState(0);
  const [freeformKey, setFreeformKey] = useState(0);

  // Restored freeform session state (passed to FreeformChatPanel)
  const [restoredSessionId, setRestoredSessionId] = useState<string | null>(null);
  const [restoredMessages, setRestoredMessages] = useState<FreeChatMessage[] | null>(null);

  /** Start a fresh chat in the current mode */
  const handleNewChat = useCallback(() => {
    if (chatMode === "rag") {
      setRagKey((k) => k + 1);
    } else {
      setFreeformKey((k) => k + 1);
      setRestoredSessionId(null);
      setRestoredMessages(null);
    }
  }, [chatMode]);

  /** Handle session restore from history panel */
  const handleRestoreSession = useCallback(
    (session: ChatSession) => {
      if (session.mode === "freeform") {
        setRestoredSessionId(session.id);
        setRestoredMessages(session.messages);
        dispatch({ type: "SET_CHAT_MODE", mode: "freeform" });
      } else {
        // RAG sessions: just switch to RAG mode
        dispatch({ type: "SET_CHAT_MODE", mode: "rag" });
      }
    },
    [dispatch],
  );

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
        const intentLabel = event.intent
          .split("_")
          .map((w: string) => w.charAt(0).toUpperCase() + w.slice(1))
          .join(" ")
          .replace("Summarize", "Summarise")
          .replace("summarize", "summarise");
        const pct = Math.round(event.confidence * 100);
        const intentMsg = event.method === "manual"
          ? `Intent: ${intentLabel}`
          : `Intent identified: ${intentLabel} (${pct}% confidence)`;
        dispatch({ type: "ADD_THINKING_STEP", message: intentMsg });
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

  return (
    <div className="relative flex flex-col h-dvh bg-background text-foreground overflow-hidden">
      {/* Background layer */}
      {theme === "beams"     && <BackgroundBeams className="absolute inset-0" />}
      {theme === "meteors"   && <Meteors className="absolute inset-0" />}
      {theme === "rain"      && <RainBackground className="absolute inset-0" />}
      {theme === "mesh"      && <MeshGradientBackground className="absolute inset-0" />}
      {theme === "paths"     && <BackgroundPaths className="absolute inset-0" />}
      {theme === "starfield" && <StarfieldBackground className="absolute inset-0" />}
      {theme === "particles" && <ParticleBackground className="absolute inset-0" />}
      {/* hidden leva panel for particle controls */}
      <Leva hidden />

        {/* ── Top bar: mode tabs + history button ─────────────────────── */}
        <header className="relative z-30 flex items-center gap-2 px-4 py-2 justify-between w-full bg-background/60 backdrop-blur-md border-b border-[#1e1e1e] shrink-0">
          <div className="flex items-center gap-2">
            {/* Mode tabs */}
            <button
              onClick={() => dispatch({ type: "SET_CHAT_MODE", mode: "rag" })}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                chatMode === "rag"
                  ? "bg-white/15 text-white border border-white/25 shadow-sm"
                  : "text-white/40 hover:text-white/70 hover:bg-white/8"
              }`}
            >
              {/* Document icon */}
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              RAG Mode
            </button>

            <button
              onClick={() => dispatch({ type: "SET_CHAT_MODE", mode: "freeform" })}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                chatMode === "freeform"
                  ? "bg-white/15 text-white border border-white/25 shadow-sm"
                  : "text-white/40 hover:text-white/70 hover:bg-white/8"
              }`}
            >
              {/* Chat bubble icon */}
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              Non-RAG Mode
            </button>
          </div>

          <div className="flex items-center gap-2">
            {/* New Chat button */}
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors text-gray-500 hover:text-gray-300 hover:bg-[#1e1e1e]"
              title="New chat"
            >
              <Plus className="w-3.5 h-3.5 shrink-0" />
              New Chat
            </button>

            {/* Theme picker */}
            <div className="relative">
              <button
                onClick={() => setThemeOpen((v) => !v)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  themeOpen
                    ? "bg-[#242424] text-gray-100 border border-[#333]"
                    : "text-gray-500 hover:text-gray-300 hover:bg-[#1e1e1e]"
                }`}
                title="Background theme"
              >
                <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4-4m0 0l4-4m-4 4h12" />
                  <circle cx="12" cy="12" r="9" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Theme
              </button>
              {themeOpen && (
                <>
                  {/* Backdrop */}
                  <div className="fixed inset-0 z-40" onClick={() => setThemeOpen(false)} />
                  <div className="absolute right-0 top-full z-50 mt-1 min-w-40 rounded-md border border-[#2a2a2a] bg-[#181818] py-1 shadow-xl">
                    {THEMES.map((t) => (
                      <button
                        key={t.id}
                        onClick={() => { setTheme(t.id); setThemeOpen(false); }}
                        className={`flex w-full items-center gap-2 px-3 py-1.5 text-xs transition-colors ${
                          theme === t.id
                            ? "text-gray-100 bg-[#242424]"
                            : "text-gray-400 hover:text-gray-200 hover:bg-[#222222]"
                        }`}
                      >
                        {theme === t.id && (
                          <svg className="w-3 h-3 shrink-0 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                        {theme !== t.id && <span className="w-3 h-3 shrink-0" />}
                        {t.label}
                      </button>
                    ))}
                  </div>
                </>
              )}

            </div>

            {/* History button */}
            <button
              onClick={() => setShowHistory((v) => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                showHistory
                  ? "bg-[#242424] text-gray-100 border border-[#333]"
                  : "text-gray-500 hover:text-gray-300 hover:bg-[#1e1e1e]"
              }`}
              title="Chat history"
            >
              <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Chat History
            </button>
          </div>
        </header>

        {/* ── Body: source panel + main + optional history ────────────── */}
        <div className="flex flex-1 min-h-0 overflow-hidden relative">

          {/* Left: Source Panel — always mounted, slides in/out via width+opacity */}
          <aside
            className="border-r shrink-0 overflow-hidden flex flex-col bg-background/70 backdrop-blur-md"
            style={{
              width:
                chatMode !== "rag" ? "0"
                : isPanelCollapsed ? "2.5rem"
                : "min(30%, 24rem)",
              minWidth:
                chatMode !== "rag" ? "0"
                : isPanelCollapsed ? "2.5rem"
                : "14rem",
              opacity: chatMode !== "rag" ? 0 : 1,
              transition: "width 280ms cubic-bezier(0.4,0,0.2,1), min-width 280ms cubic-bezier(0.4,0,0.2,1), opacity 220ms ease",
              borderRightColor: "#1e1e1e",
              pointerEvents: chatMode !== "rag" ? "none" : "auto",
            }}
          >
            {isPanelCollapsed ? (
              <div className="flex flex-col items-center pt-3">
                <button
                  onClick={() => setIsPanelCollapsed(false)}
                  className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors"
                  title="Expand sources panel"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            ) : (
              <SourcePanel
                selectedSourceIds={selectedSourceIds}
                onSelectedSourceIdsChange={setSelectedSourceIds}
                onCollapse={() => setIsPanelCollapsed(true)}
              />
            )}
          </aside>

          {/* Centre: chat panels — both always mounted, crossfade via opacity */}
          <main className="flex-1 min-w-0 h-full overflow-hidden relative">
            <div
              className="absolute inset-0"
              style={{
                opacity: chatMode === "rag" ? 1 : 0,
                pointerEvents: chatMode === "rag" ? "auto" : "none",
                transition: "opacity 220ms ease",
              }}
            >
              <RagArea
                key={ragKey}
                selectedSourceIds={selectedSourceIds}
                intentOverride={intentOverride as string}
                onData={handleData}
                onFinish={handleFinish}
              />
            </div>
            <div
              className="absolute inset-0"
              style={{
                opacity: chatMode === "freeform" ? 1 : 0,
                pointerEvents: chatMode === "freeform" ? "auto" : "none",
                transition: "opacity 220ms ease",
              }}
            >
              <FreeformChatPanel
                key={freeformKey}
                restoredSessionId={restoredSessionId}
                restoredMessages={restoredMessages}
              />
            </div>
          </main>

          {/* Right: History panel — inlined, no backdrop */}
          <HistoryPanel
            open={showHistory}
            onClose={() => setShowHistory(false)}
            onRestore={handleRestoreSession}
          />
        </div>
      </div>
  );
}
