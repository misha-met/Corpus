"use client";

import { useEffect, useRef, useState } from "react";
import {
  sourceApi,
  type CitationPayload,
  type SourceContentResponse,
  type ChunkDetailResponse,
} from "@/lib/api-client";
import { DocumentRenderer, type HighlightPayload } from "@/components/document-renderer";

interface CitationViewerModalProps {
  payload: CitationPayload;
  onClose: () => void;
}

/**
 * Modal that renders over the sources panel showing document content
 * for a clicked citation.  Accepts a full CitationPayload with optional
 * chunk_id, page_number, header_path, and chunk_text so it can:
 *
 * 1. Fetch chunk detail (if chunk_id present) for format/text.
 * 2. Fetch document content via GET /api/sources/{source_id}/content.
 * 3. Pass highlight payload to DocumentRenderer for auto-scroll + highlight.
 */
export function CitationViewerModal({
  payload,
  onClose,
}: CitationViewerModalProps) {
  const [content, setContent] = useState<SourceContentResponse | null>(null);
  const [chunkDetail, setChunkDetail] = useState<ChunkDetailResponse | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Load content (and optionally chunk detail)
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setContent(null);
    setChunkDetail(null);

    const promises: Promise<void>[] = [];

    // Always fetch document content
    promises.push(
      sourceApi
        .getContent(payload.source_id)
        .then((data) => {
          if (!cancelled) setContent(data);
        })
        .catch((err) => {
          if (!cancelled)
            setError(
              err instanceof Error ? err.message : "Failed to load content"
            );
        })
    );

    // Optionally fetch chunk detail for extra metadata / format
    if (payload.chunk_id) {
      promises.push(
        sourceApi
          .getChunk(payload.source_id, payload.chunk_id)
          .then((data) => {
            if (!cancelled) setChunkDetail(data);
          })
          .catch(() => {
            /* non-critical: we still have content */
          })
      );
    }

    Promise.all(promises).finally(() => {
      if (!cancelled) setIsLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [payload.source_id, payload.chunk_id]);

  // Focus trap & Escape to close
  useEffect(() => {
    closeBtnRef.current?.focus();

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
      // Simple focus trap: keep focus within modal
      if (e.key === "Tab") {
        const focusable = modalRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Derive format and highlight payload
  const format: "pdf" | "markdown" | "text" =
    chunkDetail?.format ?? content?.format ?? "text";

  const highlightPayload: HighlightPayload = {
    page_number: payload.page_number ?? chunkDetail?.page_number,
    header_path: payload.header_path ?? chunkDetail?.header_path,
    chunk_text: payload.chunk_text ?? chunkDetail?.chunk_text,
  };

  const hasHighlightData = !!(
    highlightPayload.chunk_text ||
    highlightPayload.header_path ||
    highlightPayload.page_number
  );

  const sourceBadgeColors: Record<string, string> = {
    original: "bg-green-900/50 text-green-300 border-green-800",
    snapshot: "bg-yellow-900/50 text-yellow-300 border-yellow-800",
    summary: "bg-blue-900/50 text-blue-300 border-blue-800",
  };

  const formatBadgeColors: Record<string, string> = {
    pdf: "bg-red-900/50 text-red-300 border-red-800",
    markdown: "bg-purple-900/50 text-purple-300 border-purple-800",
    text: "bg-gray-800 text-gray-400 border-gray-700",
  };

  return (
    <div
      className="absolute inset-0 z-50 flex flex-col bg-gray-950/95 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`Document viewer: ${payload.source_id}`}
      ref={modalRef}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          {/* PDF icon */}
          <svg
            className="w-5 h-5 text-red-400 shrink-0"
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
          <h2 className="text-sm font-medium text-gray-200 truncate">
            {payload.source_id}
          </h2>
          {content && (
            <span
              className={`shrink-0 px-2 py-0.5 text-xs rounded border ${
                sourceBadgeColors[content.content_source] ||
                "bg-gray-800 text-gray-400 border-gray-700"
              }`}
            >
              {content.content_source}
            </span>
          )}
          <span
            className={`shrink-0 px-2 py-0.5 text-xs rounded border ${
              formatBadgeColors[format] || formatBadgeColors.text
            }`}
          >
            {format.toUpperCase()}
          </span>
          {payload.page_number != null && (
            <span className="shrink-0 px-2 py-0.5 text-xs rounded border bg-gray-800 text-gray-400 border-gray-700">
              p. {payload.display_page ?? payload.page_number}
            </span>
          )}
        </div>
        <button
          ref={closeBtnRef}
          onClick={onClose}
          className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors"
          aria-label="Close document viewer"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {isLoading && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin" />
              <span className="text-sm">Loading document...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-2 px-8">
              <p className="text-sm text-red-400">{error}</p>
              <button
                onClick={() => {
                  setError(null);
                  setIsLoading(true);
                  sourceApi
                    .getContent(payload.source_id)
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
        )}

        {content && !isLoading && !error && (
          <>
            <DocumentRenderer
              content={content.content}
              format={format}
              highlight={hasHighlightData ? highlightPayload : undefined}
            />

            {/* Fallback: show "Source text:" box when we have chunk_text */}
            {highlightPayload.chunk_text && (
              <div className="mt-6 p-3 rounded-lg border border-gray-700 bg-gray-800/50">
                <p className="text-xs text-gray-400 mb-1 font-medium">
                  Source text:
                </p>
                <p className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {highlightPayload.chunk_text}
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
