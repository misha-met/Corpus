"use client";

import { useEffect, useState } from "react";
import { sourceApi, type SourceInfo } from "@/lib/api-client";

interface SourcePanelProps {
  onSelectSource: (sourceId: string) => void;
  selectedSourceId: string | null;
}

/**
 * Left panel showing the list of ingested sources.
 *
 * Features:
 * - Lists all sources with summaries
 * - Click to select and view content
 * - Delete sources
 * - Refresh button
 */
export function SourcePanel({
  onSelectSource,
  selectedSourceId,
}: SourcePanelProps) {
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function fetchSources() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await sourceApi.listSources();
      setSources(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sources");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    fetchSources();
  }, []);

  async function handleDelete(sourceId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete source "${sourceId}" and all its data?`)) return;

    try {
      await sourceApi.deleteSource(sourceId);
      setSources((prev) => prev.filter((s) => s.source_id !== sourceId));
      if (selectedSourceId === sourceId) {
        onSelectSource("");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to delete source"
      );
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Sources
        </h2>
        <button
          onClick={fetchSources}
          disabled={isLoading}
          className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors disabled:opacity-50"
          title="Refresh sources"
        >
          <svg
            className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      </div>

      {/* Source list */}
      <div className="flex-1 overflow-y-auto">
        {error && (
          <div className="px-4 py-2 text-xs text-red-400 bg-red-900/20">
            {error}
          </div>
        )}

        {isLoading && sources.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-gray-500">
            Loading sources...
          </div>
        )}

        {!isLoading && sources.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-gray-500">
            <p>No sources ingested yet.</p>
            <p className="mt-1 text-xs text-gray-600">
              Use the CLI to ingest documents
            </p>
          </div>
        )}

        {sources.map((source) => (
          <div
            key={source.source_id}
            role="button"
            tabIndex={0}
            onClick={() => onSelectSource(source.source_id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelectSource(source.source_id);
              }
            }}
            className={`w-full text-left px-4 py-3 border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors group cursor-pointer ${
              selectedSourceId === source.source_id
                ? "bg-gray-800/70 border-l-2 border-l-blue-500"
                : ""
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-gray-200 truncate">
                  {source.source_id}
                </p>
                {source.summary && (
                  <p className="mt-1 text-xs text-gray-500 line-clamp-2">
                    {source.summary}
                  </p>
                )}
              </div>
              <button
                onClick={(e) => handleDelete(source.source_id, e)}
                className="shrink-0 p-1 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                title="Delete source"
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
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-800 text-xs text-gray-600">
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </div>
    </div>
  );
}
