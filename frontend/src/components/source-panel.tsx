"use client";

import { useEffect, useState, useCallback } from "react";
import { sourceApi, type SourceInfo } from "@/lib/api-client";

interface SourcePanelProps {
  selectedSourceIds: string[];
  onSelectedSourceIdsChange: (ids: string[]) => void;
  onCollapse: () => void;
}

/**
 * Left sidebar showing ingested sources with checkboxes for query filtering.
 *
 * Features:
 * - Header with title + collapse button
 * - "Add sources" button
 * - "Select all sources" checkbox
 * - Source cards with PDF icon, truncated title, checkbox
 * - Delete source (hover action)
 * - Refresh on mount
 */
export function SourcePanel({
  selectedSourceIds,
  onSelectedSourceIdsChange,
  onCollapse,
}: SourcePanelProps) {
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSources = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await sourceApi.listSources();
      setSources(data);
      // Auto-select all on first load
      if (data.length > 0) {
        onSelectedSourceIdsChange(data.map((s) => s.source_id));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sources");
    } finally {
      setIsLoading(false);
    }
  }, [onSelectedSourceIdsChange]);

  useEffect(() => {
    fetchSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const allSelected =
    sources.length > 0 &&
    sources.every((s) => selectedSourceIds.includes(s.source_id));

  function handleSelectAll() {
    if (allSelected) {
      onSelectedSourceIdsChange([]);
    } else {
      onSelectedSourceIdsChange(sources.map((s) => s.source_id));
    }
  }

  function handleToggleSource(sourceId: string) {
    if (selectedSourceIds.includes(sourceId)) {
      onSelectedSourceIdsChange(
        selectedSourceIds.filter((id) => id !== sourceId)
      );
    } else {
      onSelectedSourceIdsChange([...selectedSourceIds, sourceId]);
    }
  }

  async function handleDelete(sourceId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete source "${sourceId}" and all its data?`)) return;

    try {
      await sourceApi.deleteSource(sourceId);
      setSources((prev) => prev.filter((s) => s.source_id !== sourceId));
      onSelectedSourceIdsChange(
        selectedSourceIds.filter((id) => id !== sourceId)
      );
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
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide">
          Sources
        </h2>
        <div className="flex items-center gap-1">
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
          <button
            onClick={onCollapse}
            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors"
            title="Collapse sidebar"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Add sources button */}
      <div className="px-4 pt-3 pb-2">
        <button
          onClick={() =>
            alert(
              "Use the CLI to ingest documents:\n\npython -m src.cli ingest <file_path>"
            )
          }
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-600 rounded-xl text-sm text-gray-300 hover:text-gray-100 transition-colors"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 4v16m8-8H4"
            />
          </svg>
          Add sources
        </button>
      </div>

      {/* Select all */}
      {sources.length > 0 && (
        <div className="px-4 py-2 border-b border-gray-800/50">
          <label className="flex items-center gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={handleSelectAll}
              className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 focus:ring-1 cursor-pointer"
            />
            <span className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors">
              Select all sources
            </span>
          </label>
        </div>
      )}

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
            <p className="mt-2 text-xs text-gray-600">
              Use the CLI to ingest documents
            </p>
          </div>
        )}

        <div className="px-3 py-1 space-y-1">
          {sources.map((source) => {
            const isChecked = selectedSourceIds.includes(source.source_id);
            return (
              <div
                key={source.source_id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-gray-800/60 transition-colors group cursor-pointer"
                onClick={() => handleToggleSource(source.source_id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    handleToggleSource(source.source_id);
                  }
                }}
              >
                {/* PDF icon */}
                <svg
                  className="w-8 h-8 text-red-400 shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
                  />
                </svg>

                {/* Title */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-200 truncate">
                    {source.source_id}
                  </p>
                  {source.summary && (
                    <p className="mt-0.5 text-xs text-gray-500 truncate">
                      {source.summary}
                    </p>
                  )}
                </div>

                {/* Delete (hover) */}
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

                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => handleToggleSource(source.source_id)}
                  onClick={(e) => e.stopPropagation()}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 focus:ring-1 cursor-pointer shrink-0"
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-800 text-xs text-gray-600">
        {sources.length} source{sources.length !== 1 ? "s" : ""} &middot;{" "}
        {selectedSourceIds.length} selected
      </div>
    </div>
  );
}
