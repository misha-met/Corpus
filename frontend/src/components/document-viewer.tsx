"use client";

import { useEffect, useState } from "react";
import { sourceApi, type SourceContentResponse } from "@/lib/api-client";

interface DocumentViewerProps {
  sourceId: string | null;
}

/**
 * Center panel showing the full text of a selected source document.
 *
 * Features:
 * - Loads content on source selection
 * - Shows content source badge (original/snapshot/summary)
 * - Graceful loading and error states
 */
export function DocumentViewer({ sourceId }: DocumentViewerProps) {
  const [content, setContent] = useState<SourceContentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sourceId) {
      setContent(null);
      setError(null);
      return;
    }

    let cancelled = false;

    async function loadContent() {
      setIsLoading(true);
      setError(null);
      try {
        const data = await sourceApi.getContent(sourceId!);
        if (!cancelled) {
          setContent(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to load content"
          );
          setContent(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadContent();
    return () => {
      cancelled = true;
    };
  }, [sourceId]);

  // Empty state
  if (!sourceId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center space-y-2">
          <svg
            className="w-12 h-12 mx-auto text-gray-700"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="text-sm">Select a source to view its content</p>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-sm">Loading document...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-2 px-8">
          <p className="text-sm text-red-400">{error}</p>
          <button
            onClick={() => {
              setError(null);
              setIsLoading(true);
              sourceApi
                .getContent(sourceId)
                .then(setContent)
                .catch((err) =>
                  setError(
                    err instanceof Error ? err.message : "Failed to load"
                  )
                )
                .finally(() => setIsLoading(false));
            }}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!content) return null;

  const sourceBadgeColors: Record<string, string> = {
    original: "bg-green-900/50 text-green-300 border-green-800",
    snapshot: "bg-yellow-900/50 text-yellow-300 border-yellow-800",
    summary: "bg-blue-900/50 text-blue-300 border-blue-800",
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <h2 className="text-sm font-medium text-gray-200 truncate">
            {content.source_id}
          </h2>
          <span
            className={`shrink-0 px-2 py-0.5 text-xs rounded border ${
              sourceBadgeColors[content.content_source] ||
              "bg-gray-800 text-gray-400 border-gray-700"
            }`}
          >
            {content.content_source}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="prose prose-invert prose-sm max-w-none">
          <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-gray-300 bg-transparent p-0 m-0">
            {content.content}
          </pre>
        </div>
      </div>
    </div>
  );
}
