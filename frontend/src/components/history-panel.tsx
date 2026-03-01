"use client";

import { useEffect, useState, useCallback } from "react";
import {
  listSessions,
  deleteSession,
  type ChatSession,
} from "@/lib/session-store";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatRelativeDate(ts: number): string {
  const now = Date.now();
  const diff = now - ts;
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface HistoryPanelProps {
  open: boolean;
  onClose: () => void;
  onRestore: (session: ChatSession) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function HistoryPanel({ open, onClose, onRestore }: HistoryPanelProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const all = await listSessions();
      setSessions(all);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load sessions whenever the panel opens
  useEffect(() => {
    if (open) {
      refresh();
    }
  }, [open, refresh]);

  const handleDelete = useCallback(
    async (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        await deleteSession(id);
        setSessions((prev) => prev.filter((s) => s.id !== id));
      } catch (err) {
        console.error("Failed to delete session:", err);
      }
    },
    [],
  );

  const filteredSessions = sessions.filter((s) => {
    if (!search.trim()) return true;
    const q = search.toLowerCase();
    return (
      s.title.toLowerCase().includes(q) ||
      s.messages.some((m) => m.content.toLowerCase().includes(q))
    );
  });

  if (!open) return null;

  return (
    <aside
      className="flex flex-col w-72 shrink-0 overflow-hidden backdrop-blur-xl"
      style={{
        background: "rgba(8,8,8,0.88)",
        borderLeft: "1px solid rgba(255,255,255,0.10)",
        boxShadow: "-4px 0 24px rgba(0,0,0,0.5), inset 1px 0 0 rgba(255,255,255,0.04)",
      }}
    >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <h2 className="text-sm font-semibold text-foreground tracking-wide">
            Chat History
          </h2>
          <button
            onClick={onClose}
            className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent rounded transition-colors"
            title="Close"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Search */}
        <div className="px-4 py-2 shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div className="relative">
            <svg
              className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search conversations..."
              className="w-full rounded-lg pl-8 pr-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none transition-colors"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.10)",
                boxShadow: "0 1px 4px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03)",
              }}
            />
          </div>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto py-2">
          {isLoading ? (
            <div className="flex items-center justify-center py-12 text-gray-600 text-sm">
              Loading…
            </div>
          ) : filteredSessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 gap-2 text-center px-6">
              <svg className="w-8 h-8 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className="text-gray-600 text-xs">
                {search ? "No matching conversations" : "No saved conversations yet"}
              </p>
            </div>
          ) : (
            <ul className="divide-y divide-[#1e1e1e]">
              {filteredSessions.map((session) => (
                <li key={session.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => { onRestore(session); onClose(); }}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRestore(session); onClose(); } }}
                    className="w-full text-left px-4 py-3 hover:bg-accent transition-colors group relative cursor-pointer"
                  >
                    {/* Mode badge */}
                    <div className="flex items-center gap-2 mb-1">
                      <span
              className="text-[9px] font-medium px-1.5 py-0.5 rounded-full text-gray-400"
              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)" }}
            >
                        {session.mode === "freeform" ? "Non-RAG" : "RAG"}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {formatRelativeDate(session.updatedAt)}
                      </span>
                    </div>

                    {/* Title */}
                    <p className="text-xs text-foreground leading-snug line-clamp-2 pr-8">
                      {session.title}
                    </p>

                    {/* Message count */}
                    {session.messages.length > 0 && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {Math.floor(session.messages.length / 2)} exchange
                        {Math.floor(session.messages.length / 2) !== 1 ? "s" : ""}
                      </p>
                    )}

                    {/* Delete button */}
                    <button
                      onClick={(e) => handleDelete(session.id, e)}
                      className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 p-1 text-gray-600 hover:text-red-400 hover:bg-red-900/20 rounded transition-all"
                      title="Delete session"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        {sessions.length > 0 && (
          <div className="px-4 py-2 shrink-0" style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            <p className="text-[10px] text-muted-foreground text-center" style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}>
              {sessions.length} saved conversation{sessions.length !== 1 ? "s" : ""}
            </p>
          </div>
        )}
      </aside>
  );
}
