"use client";

import { useState, useCallback } from "react";
import { ChatPanel } from "@/components/chat-panel";
import { SourcePanel } from "@/components/source-panel";
import { CitationViewerModal } from "@/components/citation-viewer-modal";
import { AppProvider } from "@/context/app-context";
import type { CitationPayload } from "@/lib/api-client";

/**
 * Two-panel layout:
 *
 * ┌──────────────┬────────────────────────────────┐
 * │   Sources    │            Chat                │
 * │  (~30%)      │           (~70%)               │
 * └──────────────┴────────────────────────────────┘
 *
 * - Left: Source sidebar with checkboxes, add sources, select all
 * - Right: Chat with citations, actions, timestamp
 * - Citation click opens a modal over the sources panel
 */
function AppContent() {
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [citationModal, setCitationModal] = useState<{
    open: boolean;
    payload: CitationPayload;
  }>({ open: false, payload: { source_id: "" } });

  const handleCitationClick = useCallback((payload: CitationPayload) => {
    setCitationModal({ open: true, payload });
  }, []);

  const handleCloseCitationModal = useCallback(() => {
    setCitationModal({ open: false, payload: { source_id: "" } });
  }, []);

  return (
    <div className="h-screen flex bg-gray-950 text-gray-100">
      {/* Left: Source sidebar */}
      {!sidebarCollapsed && (
        <aside className="w-[30%] min-w-70 max-w-90 border-r border-gray-800 bg-gray-900/30 shrink-0 relative">
          <SourcePanel
            selectedSourceIds={selectedSourceIds}
            onSelectedSourceIdsChange={setSelectedSourceIds}
            onCollapse={() => setSidebarCollapsed(true)}
          />
          {/* Citation modal renders over sources panel */}
          {citationModal.open && (
            <CitationViewerModal
              payload={citationModal.payload}
              onClose={handleCloseCitationModal}
            />
          )}
        </aside>
      )}

      {/* Collapsed sidebar indicator */}
      {sidebarCollapsed && (
        <div className="w-12 border-r border-gray-800 bg-gray-900/30 shrink-0 flex flex-col items-center pt-3">
          <button
            onClick={() => setSidebarCollapsed(false)}
            className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors"
            title="Expand sources"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      )}

      {/* Right: Chat panel */}
      <main className="flex-1 min-w-0">
        <ChatPanel
          selectedSourceIds={selectedSourceIds}
          sourceCount={selectedSourceIds.length}
          onCitationClick={handleCitationClick}
        />
      </main>
    </div>
  );
}

export default function Page() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
