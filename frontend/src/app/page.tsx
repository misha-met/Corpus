"use client";

import { useEffect, useCallback, useState, useMemo, useRef } from "react";
import { FreeformChatPanel } from "@/components/freeform-chat-panel";
import { HistoryPanel } from "@/components/history-panel";
import { CorpusMap } from "@/components/map/CorpusMap";
import { PeopleDictionary } from "@/components/people/PeopleDictionary";
import { useAppDispatch, useAppState } from "@/context/app-context";
import type { ChatSession } from "@/lib/session-store";
import type { FreeChatMessage } from "@/lib/session-store";
import { Leva } from "leva";
import { useTheme } from "@/context/theme-context";
import { BackgroundLayer } from "@/components/layout/background-layer";
import { OverlayPanel } from "@/components/layout/overlay-panel";
import { TopBar } from "@/components/layout/top-bar";
import { SourcePanelContainer } from "@/components/layout/source-panel-container";
import { RagArea } from "@/components/layout/rag-area";
import { clamp } from "@/lib/clamp";
import { GLASS, DARK_BG_THEMES, type ChromeStyles } from "@/lib/theme-constants";
import { usePersistedState } from "@/hooks/use-persisted-state";
import { useStreamHandler } from "@/hooks/use-stream-handler";
import { ErrorBoundary } from "@/components/error-boundary";

export default function Page() {
  const dispatch = useAppDispatch();
  const { intentOverride, chatMode } = useAppState();
  const { theme, setTheme } = useTheme();

  const glass = useMemo(() => GLASS[theme], [theme]);

  // Source panel collapse state
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // Source IDs selected in the panel (passed down for UI checkbox state)
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);

  const clampMap = useCallback((v: number) => clamp(v, 0.5, 0.99, 0.75), []);
  const [mapThreshold, setMapThreshold] = usePersistedState("dh-map-threshold-v1", 0.75, clampMap);

  // Debounced Map Threshold
  const [localMapThreshold, setLocalMapThreshold] = useState(mapThreshold);
  const mapThresholdTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleLocalMapThresholdChange = useCallback((val: number) => {
    setLocalMapThreshold(val);
    if (mapThresholdTimeoutRef.current) clearTimeout(mapThresholdTimeoutRef.current);
    mapThresholdTimeoutRef.current = setTimeout(() => {
      setMapThreshold(val);
    }, 150);
  }, [setMapThreshold]);

  useEffect(() => {
    setLocalMapThreshold(mapThreshold);
  }, [mapThreshold]);

  // History panel
  const [showHistory, setShowHistory] = useState(false);
  const [isMapOpen, setIsMapOpen] = useState(false);
  const [isPeopleOpen, setIsPeopleOpen] = useState(false);
  const [mapRefreshNonce, setMapRefreshNonce] = useState(0);
  const [peopleRefreshNonce, setPeopleRefreshNonce] = useState(0);

  useEffect(() => {
    dispatch({ type: "SET_SELECTED_SOURCE_IDS", sourceIds: selectedSourceIds });
  }, [dispatch, selectedSourceIds]);

  const chromeStyles = useMemo<ChromeStyles>(() => {
    const isOverlayOpen = isMapOpen || isPeopleOpen;
    return {
      bg: isOverlayOpen ? "rgba(0,0,0,0.58)" : glass.bg,
      backdrop: isOverlayOpen ? "blur(16px) saturate(115%)" : glass.backdrop,
      borderColor: isOverlayOpen ? "rgba(255,255,255,0.14)" : glass.border,
      fadeOpacity: isOverlayOpen ? 0.9 : 0,
    };
  }, [isMapOpen, isPeopleOpen, glass]);

  // New chat reset keys
  const [ragKey, setRagKey] = useState(0);
  const [freeformKey, setFreeformKey] = useState(0);

  // Restored freeform session state
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
        dispatch({ type: "SET_CHAT_MODE", mode: "rag" });
      }
    },
    [dispatch],
  );

  const { handleData, handleFinish } = useStreamHandler(dispatch);

  const handleToggleMap = useCallback(() => {
    setIsMapOpen((v) => {
      const next = !v;
      if (next) setIsPeopleOpen(false);
      return next;
    });
  }, []);

  const handleTogglePeople = useCallback(() => {
    setIsPeopleOpen((v) => {
      const next = !v;
      if (next) setIsMapOpen(false);
      return next;
    });
  }, []);

  const handleToggleHistory = useCallback(() => {
    setShowHistory((v) => !v);
  }, []);

  const handleCollapse = useCallback(() => setIsPanelCollapsed(true), []);
  const handleExpand = useCallback(() => setIsPanelCollapsed(false), []);
  const handleSourcesChanged = useCallback(() => {
    setMapRefreshNonce((n) => n + 1);
    setPeopleRefreshNonce((n) => n + 1);
  }, []);

  const handleSetMode = useCallback((mode: "rag" | "freeform") => {
    dispatch({ type: "SET_CHAT_MODE", mode });
  }, [dispatch]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (isMapOpen) { setIsMapOpen(false); return; }
      if (isPeopleOpen) { setIsPeopleOpen(false); return; }
      if (showHistory) { setShowHistory(false); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isMapOpen, isPeopleOpen, showHistory]);

  const [tabVisible, setTabVisible] = useState(true);
  useEffect(() => {
    const handler = () => setTabVisible(!document.hidden);
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);
  const bgPaused = !tabVisible || isMapOpen || isPeopleOpen;

  return (
    <div
      className="relative flex flex-col h-dvh text-foreground overflow-hidden"
      style={{
        background: DARK_BG_THEMES.has(theme) ? "#0a0a0a" : "var(--background)",
      }}
    >
      <ErrorBoundary>
        <BackgroundLayer theme={theme} paused={bgPaused} />
      </ErrorBoundary>
      <Leva hidden />

      <TopBar
        chatMode={chatMode}
        isMapOpen={isMapOpen}
        isPeopleOpen={isPeopleOpen}
        showHistory={showHistory}
        theme={theme}
        chromeStyles={chromeStyles}
        onSetMode={handleSetMode}
        onToggleMap={handleToggleMap}
        onTogglePeople={handleTogglePeople}
        onNewChat={handleNewChat}
        onSetTheme={setTheme}
        onToggleHistory={handleToggleHistory}
      />

      <div className="flex flex-1 min-h-0 overflow-hidden relative">
        <SourcePanelContainer
          chatMode={chatMode}
          isPanelCollapsed={isPanelCollapsed}
          chromeStyles={chromeStyles}
          selectedSourceIds={selectedSourceIds}
          onSelectedSourceIdsChange={setSelectedSourceIds}
          onCollapse={handleCollapse}
          onExpand={handleExpand}
          onSourcesChanged={handleSourcesChanged}
        />

        <main className="flex-1 min-w-0 h-full overflow-hidden relative">
          <div
            className="absolute inset-0"
            style={{
              opacity: chatMode === "rag" ? 1 : 0,
              pointerEvents: chatMode === "rag" ? "auto" : "none",
              transition: "opacity 220ms ease",
            }}
          >
            <ErrorBoundary>
              <RagArea
                key={ragKey}
                selectedSourceIds={selectedSourceIds}
                intentOverride={intentOverride as string}
                onData={handleData}
                onFinish={handleFinish}
              />
            </ErrorBoundary>
          </div>
          <div
            className="absolute inset-0"
            style={{
              opacity: chatMode === "freeform" ? 1 : 0,
              pointerEvents: chatMode === "freeform" ? "auto" : "none",
              transition: "opacity 220ms ease",
            }}
          >
            <ErrorBoundary>
              <FreeformChatPanel
                key={freeformKey}
                restoredSessionId={restoredSessionId}
                restoredMessages={restoredMessages}
              />
            </ErrorBoundary>
          </div>

          <OverlayPanel open={isMapOpen} onClose={() => setIsMapOpen(false)} title="map">
            <div className="absolute top-3 left-3 z-30 w-[min(78vw,18rem)] rounded-xl border border-white/20 bg-black/72 px-3 py-2.5 text-white shadow-lg backdrop-blur-md">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/75">Map confidence</p>
              <input
                type="range"
                min={0.5}
                max={0.99}
                step={0.01}
                value={localMapThreshold}
                onChange={(event) => handleLocalMapThresholdChange(Number(event.target.value))}
                className="mt-2 w-full accent-amber-400"
                aria-label="Map confidence threshold"
              />
              <div className="mt-1.5 flex items-center justify-between text-[11px] text-white/70">
                <span>{Math.round(localMapThreshold * 100)}% minimum</span>
                <span>{selectedSourceIds.length} sources</span>
              </div>
            </div>

            <ErrorBoundary>
              <CorpusMap
                active={isMapOpen}
                refreshNonce={mapRefreshNonce}
                threshold={mapThreshold}
                sourceIds={selectedSourceIds}
              />
            </ErrorBoundary>
          </OverlayPanel>

          <OverlayPanel
            open={isPeopleOpen}
            onClose={() => setIsPeopleOpen(false)}
            title="people dictionary"
          >
            <PeopleDictionary
              active={isPeopleOpen}
              refreshNonce={peopleRefreshNonce}
              sourceIds={selectedSourceIds}
            />
          </OverlayPanel>
        </main>

        <HistoryPanel
          open={showHistory}
          onClose={() => setShowHistory(false)}
          onRestore={handleRestoreSession}
          panelBg={glass.bg}
          panelBackdrop={glass.backdrop}
          panelBorderColor={glass.border}
        />
      </div>
    </div>
  );
}
