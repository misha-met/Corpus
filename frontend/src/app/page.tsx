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
import { CorpusMap } from "@/components/map/CorpusMap";
import { PeopleDictionary } from "@/components/people/PeopleDictionary";
import { useAppDispatch, useAppState } from "@/context/app-context";
import { parseCustomEvent } from "@/lib/event-parser";
import type { ChatSession } from "@/lib/session-store";
import type { FreeChatMessage } from "@/lib/session-store";
import { Map as MapIcon, PaletteIcon, Plus, UserRound, X } from "lucide-react";
import {
  PickerRoot,
  PickerTrigger,
  PickerContent,
  PickerItem,
} from "@/components/ui/picker";
import { Meteors } from "@/components/ui/meteors";
import { RainBackground } from "@/components/ui/rain";
import { MeshGradientBackground } from "@/components/ui/mesh-gradient";
import { StarfieldBackground } from "@/components/ui/starfield";
import { ParticleBackground } from "@/components/ui/particles";
import { StarsBackground } from "@/components/ui/stars-background";
import { DarkVeilBackground } from "@/components/ui/dark-veil";
import { Leva } from "leva";
import { useTheme, type BackgroundTheme } from "@/context/theme-context";

const MAP_THRESHOLD_STORAGE_KEY = "dh-map-threshold-v1";
const DEFAULT_MAP_THRESHOLD = 0.75;
const PEOPLE_THRESHOLD_STORAGE_KEY = "dh-people-threshold-v1";
const DEFAULT_PEOPLE_THRESHOLD = 0.7;

function clampMapThreshold(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_MAP_THRESHOLD;
  return Math.max(0.5, Math.min(0.99, value));
}

function clampPeopleThreshold(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_PEOPLE_THRESHOLD;
  return Math.max(0.3, Math.min(0.99, value));
}

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

  const GLASS: Record<BackgroundTheme, { bg: string; backdrop: string; border: string }> = {
    meteors:   { bg: "rgba(255,255,255,0.0)", backdrop: "blur(7px)",                  border: "rgba(255,255,255,0.060)" },
    rain:      { bg: "rgba(0,0,0,0.175)",     backdrop: "blur(9px) saturate(100%)",   border: "rgba(255,255,255,0.000)" },
    mesh:      { bg: "rgba(0,0,0,0.055)",     backdrop: "blur(9px) saturate(110%)",   border: "rgba(255,255,255,0.000)" },
    starfield: { bg: "rgba(0,0,0,0.165)",     backdrop: "blur(10px) saturate(100%)",  border: "rgba(255,255,255,0.000)" },
    particles: { bg: "rgba(255,255,255,0.030)", backdrop: "blur(4px)",                border: "rgba(255,255,255,0.12)"  },
    stars:     { bg: "rgba(0,0,0,0.145)",     backdrop: "blur(4px) saturate(100%)",   border: "rgba(255,255,255,0.000)" },
    darkveil:  { bg: "rgba(0,0,0,0.250)",     backdrop: "blur(6px) saturate(110%)",   border: "rgba(255,255,255,0.000)" },
  };
  const glass = GLASS[theme];
  const panelBg = glass.bg;
  const panelBackdrop = glass.backdrop;
  const panelBorderColor = glass.border;

  const THEMES: { id: BackgroundTheme; label: string }[] = [
    { id: "stars",     label: "Stars" },
    { id: "meteors",   label: "Meteors" },
    { id: "rain",      label: "Rain" },
    { id: "mesh",      label: "Gradient" },
    { id: "starfield", label: "Starfield" },
    { id: "particles", label: "Particles" },
    { id: "darkveil",  label: "Dark Veil" },
  ];

  // Source panel collapse state
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // Source IDs selected in the panel (passed down for UI checkbox state)
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [mapThreshold, setMapThreshold] = useState<number>(DEFAULT_MAP_THRESHOLD);
  const [isMapThresholdHydrated, setIsMapThresholdHydrated] = useState(false);
  const [peopleThreshold, setPeopleThreshold] = useState<number>(DEFAULT_PEOPLE_THRESHOLD);
  const [isPeopleThresholdHydrated, setIsPeopleThresholdHydrated] = useState(false);

  // History panel
  const [showHistory, setShowHistory] = useState(false);
  const [isMapOpen, setIsMapOpen] = useState(false);
  const [isPeopleOpen, setIsPeopleOpen] = useState(false);
  const [mapRefreshNonce, setMapRefreshNonce] = useState(0);
  const [peopleRefreshNonce, setPeopleRefreshNonce] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(MAP_THRESHOLD_STORAGE_KEY);
      if (raw) {
        setMapThreshold(clampMapThreshold(Number(raw)));
      }
    } finally {
      setIsMapThresholdHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(PEOPLE_THRESHOLD_STORAGE_KEY);
      if (raw) {
        setPeopleThreshold(clampPeopleThreshold(Number(raw)));
      }
    } finally {
      setIsPeopleThresholdHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !isMapThresholdHydrated) return;
    window.localStorage.setItem(MAP_THRESHOLD_STORAGE_KEY, String(mapThreshold));
  }, [mapThreshold, isMapThresholdHydrated]);

  useEffect(() => {
    if (typeof window === "undefined" || !isPeopleThresholdHydrated) return;
    window.localStorage.setItem(PEOPLE_THRESHOLD_STORAGE_KEY, String(peopleThreshold));
  }, [peopleThreshold, isPeopleThresholdHydrated]);

  useEffect(() => {
    dispatch({ type: "SET_SELECTED_SOURCE_IDS", sourceIds: selectedSourceIds });
  }, [dispatch, selectedSourceIds]);

  const isOverlayOpen = isMapOpen || isPeopleOpen;

  const chromeBg = isOverlayOpen ? "rgba(0,0,0,0.58)" : panelBg;
  const chromeBackdrop = isOverlayOpen ? "blur(16px) saturate(115%)" : panelBackdrop;
  const chromeBorderColor = isOverlayOpen ? "rgba(255,255,255,0.14)" : panelBorderColor;
  const chromeFadeOpacity = isOverlayOpen ? 0.9 : 0;
  const mapOverlayTransform = isMapOpen ? "translateY(0px) scale(1)" : "translateY(10px) scale(0.992)";
  const mapOverlayFilter = isMapOpen ? "blur(0px)" : "blur(1.75px)";
  const mapScrimOpacity = isMapOpen ? 1 : 0;
  const mapCardOpacity = isMapOpen ? 1 : 0;
  const mapCardTransform = isMapOpen ? "translateY(0px) scale(1)" : "translateY(18px) scale(0.972)";
  const peopleOverlayTransform = isPeopleOpen ? "translateY(0px) scale(1)" : "translateY(10px) scale(0.992)";
  const peopleOverlayFilter = isPeopleOpen ? "blur(0px)" : "blur(1.75px)";
  const peopleScrimOpacity = isPeopleOpen ? 1 : 0;
  const peopleCardOpacity = isPeopleOpen ? 1 : 0;
  const peopleCardTransform = isPeopleOpen ? "translateY(0px) scale(1)" : "translateY(18px) scale(0.972)";

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
      case "trace-id":
        dispatch({
          type: "SET_TRACE_INFO",
          traceId: event.traceId,
          spanId: event.spanId,
        });
        break;
      // "finish-step" and other internal protocol events are ignored here
    }
  }, [dispatch]);

  /**
   * onFinish — fires once when the entire stream completes.
   * Clears the statusMessage and resets the stream-started guard for the
   * next query.
   */
  const handleFinish = useCallback(() => {
    streamStartedRef.current = false;
    dispatch({ type: "QUERY_FINISHED" });
  }, [dispatch]);

  return (
    <div
      className="relative flex flex-col h-dvh text-foreground overflow-hidden"
      style={{ background: (theme === "meteors" || theme === "stars" || theme === "starfield" || theme === "darkveil") ? "#0a0a0a" : "var(--background)" }}
    >
      {/* Background layer */}
      {theme === "meteors"   && <Meteors className="absolute inset-0" />}
      {theme === "rain"      && <RainBackground className="absolute inset-0" />}
      {theme === "mesh"      && <MeshGradientBackground className="absolute inset-0" />}
      {theme === "starfield" && <StarfieldBackground className="absolute inset-0" />}
      {theme === "particles" && <ParticleBackground className="absolute inset-0" />}
      {theme === "stars"     && <StarsBackground className="absolute inset-0" />}
      {theme === "darkveil"  && <DarkVeilBackground className="absolute inset-0" />}
      {/* hidden leva panel for particle controls */}
      <Leva hidden />

        {/* ── Top bar: mode tabs + history button ─────────────────────── */}
        <header
          className="relative z-30 flex items-center gap-2 px-4 py-2 justify-between w-full shrink-0"
          style={{
            background: chromeBg,
            borderBottom: `1px solid ${chromeBorderColor}`,
            boxShadow: "0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35), inset 0 -1px 0 rgba(255,255,255,0.04)",
            backdropFilter: chromeBackdrop,
            WebkitBackdropFilter: chromeBackdrop,
            transition: "background 520ms cubic-bezier(0.22,1,0.36,1), border-color 520ms cubic-bezier(0.22,1,0.36,1), backdrop-filter 520ms cubic-bezier(0.22,1,0.36,1)",
            isolation: "isolate",
          }}
        >
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background: "linear-gradient(180deg, rgba(0,0,0,0.58) 0%, rgba(0,0,0,0.72) 100%)",
              opacity: chromeFadeOpacity,
              transition: "opacity 560ms cubic-bezier(0.22,1,0.36,1)",
            }}
          />

          <div className="relative z-10 flex items-center gap-2">
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

            <button
              onClick={() => {
                setIsMapOpen((v) => {
                  const next = !v;
                  if (next) setIsPeopleOpen(false);
                  return next;
                });
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                isMapOpen
                  ? "bg-white/15 text-white border border-white/25 shadow-sm"
                  : "text-white/40 hover:text-white/70 hover:bg-white/8"
              }`}
              title="Open map"
            >
              <MapIcon className="w-3.5 h-3.5 shrink-0" />
              Map
            </button>

            <button
              onClick={() => {
                setIsPeopleOpen((v) => {
                  const next = !v;
                  if (next) setIsMapOpen(false);
                  return next;
                });
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                isPeopleOpen
                  ? "bg-white/15 text-white border border-white/25 shadow-sm"
                  : "text-white/40 hover:text-white/70 hover:bg-white/8"
              }`}
              title="Open people dictionary"
            >
              <UserRound className="w-3.5 h-3.5 shrink-0" />
              People
            </button>
          </div>

          <div className="relative z-10 flex items-center gap-2">
            {/* New Chat button */}
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors text-white/40 hover:text-white/70 hover:bg-white/8"
              title="New chat"
            >
              <Plus className="w-3.5 h-3.5 shrink-0" />
              New Chat
            </button>

            {/* Theme picker */}
            <PickerRoot open={themeOpen} onOpenChange={setThemeOpen}>
              <PickerTrigger
                variant="ghost"
                size="sm"
                className="text-white/40 hover:text-white/70 hover:bg-white/8 data-[state=open]:bg-white/15 data-[state=open]:text-white data-[state=open]:border data-[state=open]:border-white/25"
                title="Background theme"
              >
                <PaletteIcon className="w-3.5 h-3.5 shrink-0" />
                Theme
              </PickerTrigger>
              <PickerContent align="end" className="min-w-36">
                {THEMES.map((t) => (
                  <PickerItem
                    key={t.id}
                    selected={theme === t.id}
                    onClick={() => { setTheme(t.id); setThemeOpen(false); }}
                  >
                    {t.label}
                  </PickerItem>
                ))}
              </PickerContent>
            </PickerRoot>

            <button
              onClick={() => setShowHistory((v) => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                showHistory
                  ? "bg-white/15 text-white border border-white/25 shadow-sm"
                  : "text-white/40 hover:text-white/70 hover:bg-white/8"
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
            className="relative shrink-0 overflow-hidden flex flex-col"
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
              transition: "width 280ms cubic-bezier(0.4,0,0.2,1), min-width 280ms cubic-bezier(0.4,0,0.2,1), opacity 240ms ease, background 520ms cubic-bezier(0.22,1,0.36,1), border-color 520ms cubic-bezier(0.22,1,0.36,1), backdrop-filter 520ms cubic-bezier(0.22,1,0.36,1)",
              background: chromeBg,
              borderRight: `1px solid ${chromeBorderColor}`,
              boxShadow: "2px 0 8px rgba(0,0,0,0.45), 1px 0 2px rgba(0,0,0,0.35), inset -1px 0 0 rgba(255,255,255,0.04)",
              backdropFilter: chromeBackdrop,
              WebkitBackdropFilter: chromeBackdrop,
              willChange: "opacity, background, backdrop-filter",
              isolation: "isolate",
              pointerEvents: chatMode !== "rag" ? "none" : "auto",
            }}
          >
            <div
              className="pointer-events-none absolute inset-0"
              style={{
                background: "linear-gradient(180deg, rgba(0,0,0,0.62) 0%, rgba(0,0,0,0.68) 100%)",
                opacity: chromeFadeOpacity,
                transition: "opacity 620ms cubic-bezier(0.22,1,0.36,1)",
              }}
            />

            <div className="relative z-10 flex h-full flex-col">
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
                  onSourcesChanged={() => {
                    setMapRefreshNonce((n) => n + 1);
                    setPeopleRefreshNonce((n) => n + 1);
                  }}
                />
              )}
            </div>
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

            <div
              className="absolute inset-0 z-20 p-2 sm:p-3 md:p-4"
              style={{
                opacity: isMapOpen ? 1 : 0,
                transform: mapOverlayTransform,
                filter: mapOverlayFilter,
                pointerEvents: isMapOpen ? "auto" : "none",
                transition: "opacity 420ms cubic-bezier(0.22,1,0.36,1), transform 520ms cubic-bezier(0.22,1,0.36,1), filter 420ms ease",
                willChange: "opacity, transform, filter",
                transformOrigin: "center center",
              }}
            >
              <div
                className="pointer-events-none absolute inset-0"
                style={{
                  background:
                    "linear-gradient(180deg, rgba(7,10,14,0.82) 0%, rgba(7,10,14,0.90) 100%), radial-gradient(120% 80% at 50% 50%, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.00) 68%)",
                  opacity: mapScrimOpacity,
                  transition: "opacity 520ms cubic-bezier(0.22,1,0.36,1)",
                }}
              />

              <div
                className="relative h-full w-full"
                style={{
                  opacity: mapCardOpacity,
                  transform: mapCardTransform,
                  transition: "opacity 460ms cubic-bezier(0.22,1,0.36,1), transform 560ms cubic-bezier(0.22,1,0.36,1)",
                  willChange: "opacity, transform",
                }}
              >
                <div className="relative h-full w-full overflow-hidden rounded-2xl border border-white/20 bg-black/92 shadow-[0_20px_60px_rgba(0,0,0,0.66),inset_0_1px_0_rgba(255,255,255,0.08)]">
                  <div className="pointer-events-none absolute inset-x-0 top-0 z-20 h-16 bg-gradient-to-b from-black/65 via-black/30 to-transparent" />

                  <div className="absolute top-3 left-3 z-30 w-[min(78vw,18rem)] rounded-xl border border-white/20 bg-black/72 px-3 py-2.5 text-white shadow-lg backdrop-blur-md">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/75">Map confidence</p>
                    <input
                      type="range"
                      min={0.5}
                      max={0.99}
                      step={0.01}
                      value={mapThreshold}
                      onChange={(event) => setMapThreshold(clampMapThreshold(Number(event.target.value)))}
                      className="mt-2 w-full accent-amber-400"
                      aria-label="Map confidence threshold"
                    />
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-white/70">
                      <span>{Math.round(mapThreshold * 100)}% minimum</span>
                      <span>{selectedSourceIds.length} sources</span>
                    </div>
                  </div>

                  <button
                    onClick={() => setIsMapOpen(false)}
                    className="absolute top-3 right-3 z-30 p-2 rounded-full bg-black/75 text-white/80 hover:text-white hover:bg-black border border-white/20 shadow-md"
                    title="Close map"
                  >
                    <X className="w-4 h-4" />
                  </button>

                  <CorpusMap
                    active={isMapOpen}
                    refreshNonce={mapRefreshNonce}
                    threshold={mapThreshold}
                    sourceIds={selectedSourceIds}
                  />
                </div>
              </div>
            </div>

            <div
              className="absolute inset-0 z-20 p-2 sm:p-3 md:p-4"
              style={{
                opacity: isPeopleOpen ? 1 : 0,
                transform: peopleOverlayTransform,
                filter: peopleOverlayFilter,
                pointerEvents: isPeopleOpen ? "auto" : "none",
                transition: "opacity 420ms cubic-bezier(0.22,1,0.36,1), transform 520ms cubic-bezier(0.22,1,0.36,1), filter 420ms ease",
                willChange: "opacity, transform, filter",
                transformOrigin: "center center",
              }}
            >
              <div
                className="pointer-events-none absolute inset-0"
                style={{
                  background:
                    "linear-gradient(180deg, rgba(12,9,4,0.84) 0%, rgba(12,9,4,0.92) 100%), radial-gradient(120% 80% at 50% 50%, rgba(251,191,36,0.05) 0%, rgba(251,191,36,0.00) 68%)",
                  opacity: peopleScrimOpacity,
                  transition: "opacity 520ms cubic-bezier(0.22,1,0.36,1)",
                }}
              />

              <div
                className="relative h-full w-full"
                style={{
                  opacity: peopleCardOpacity,
                  transform: peopleCardTransform,
                  transition: "opacity 460ms cubic-bezier(0.22,1,0.36,1), transform 560ms cubic-bezier(0.22,1,0.36,1)",
                  willChange: "opacity, transform",
                }}
              >
                <div className="relative h-full w-full overflow-hidden rounded-2xl border border-white/20 bg-black/92 shadow-[0_20px_60px_rgba(0,0,0,0.66),inset_0_1px_0_rgba(255,255,255,0.08)]">
                  <div className="pointer-events-none absolute inset-x-0 top-0 z-20 h-16 bg-gradient-to-b from-black/65 via-black/30 to-transparent" />

                  <button
                    onClick={() => setIsPeopleOpen(false)}
                    className="absolute top-3 right-3 z-30 p-2 rounded-full bg-black/75 text-white/80 hover:text-white hover:bg-black border border-white/20 shadow-md"
                    title="Close people dictionary"
                  >
                    <X className="w-4 h-4" />
                  </button>

                  <PeopleDictionary
                    active={isPeopleOpen}
                    refreshNonce={peopleRefreshNonce}
                    threshold={peopleThreshold}
                    sourceIds={selectedSourceIds}
                    onThresholdChange={(value) => setPeopleThreshold(clampPeopleThreshold(value))}
                  />
                </div>
              </div>
            </div>
          </main>

          {/* Right: History panel — inlined, no backdrop */}
          <HistoryPanel
            open={showHistory}
            onClose={() => setShowHistory(false)}
            onRestore={handleRestoreSession}
            panelBg={panelBg}
            panelBackdrop={panelBackdrop}
            panelBorderColor={panelBorderColor}
          />
        </div>


      </div>
  );
}
