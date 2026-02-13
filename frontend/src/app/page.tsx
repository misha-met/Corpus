"use client";

import { useState } from "react";
import { ChatPanel } from "@/components/chat-panel";
import { DocumentViewer } from "@/components/document-viewer";
import { SourcePanel } from "@/components/source-panel";
import { AppProvider, useAppState } from "@/context/app-context";

/**
 * Main 3-panel layout inspired by NotebookLM:
 *
 * ┌──────────┬────────────────────┬──────────────┐
 * │  Sources │  Document Viewer   │    Chat      │
 * │  (left)  │     (center)       │   (right)    │
 * └──────────┴────────────────────┴──────────────┘
 *
 * - Left: Source list with summaries and delete
 * - Center: Full document text viewer
 * - Right: AI chat interface
 */
function AppContent() {
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const { lastSources } = useAppState();

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-2.5 border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold tracking-tight">DH Notebook</h1>
          <span className="text-xs text-gray-500 font-mono px-1.5 py-0.5 bg-gray-800/50 rounded">
            offline
          </span>
        </div>
        {lastSources.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <span className="text-gray-600">Retrieved from:</span>
            {lastSources.map((id) => (
              <button
                key={id}
                onClick={() => setSelectedSourceId(id)}
                className="px-2 py-0.5 bg-gray-800 rounded-md border border-gray-700 hover:border-blue-600 hover:text-blue-400 transition-colors cursor-pointer"
              >
                {id}
              </button>
            ))}
          </div>
        )}
      </header>

      {/* 3-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Source panel */}
        <aside className="w-64 border-r border-gray-800 bg-gray-900/30 shrink-0">
          <SourcePanel
            onSelectSource={setSelectedSourceId}
            selectedSourceId={selectedSourceId}
          />
        </aside>

        {/* Center: Document viewer */}
        <section className="flex-1 border-r border-gray-800 bg-gray-900/10 min-w-0">
          <DocumentViewer sourceId={selectedSourceId} />
        </section>

        {/* Right: Chat panel */}
        <aside className="w-[420px] shrink-0">
          <ChatPanel />
        </aside>
      </div>
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
