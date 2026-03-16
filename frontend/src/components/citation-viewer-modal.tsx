"use client";

import { useEffect, useState } from "react";
import {
  sourceApi,
  type SourceContentResponse,
  type ChunkDetailResponse,
} from "@/lib/api-client";
import { DocumentRenderer, type HighlightPayload } from "@/components/document-renderer";
import { useAppState, useAppDispatch } from "@/context/app-context";

/**
 * CitationPanelReader — panel-embedded citation viewer.
 *
 * Renders within the SourcePanel bounds (flex-col, h-full) rather than as a
 * full-screen modal overlay.  Reads activeCitation from AppContext and closes
 * by dispatching SET_ACTIVE_CITATION(null).
 */
export function CitationPanelReader() {
  const { activeCitation } = useAppState();

  if (activeCitation === null) return null;

  return (
    <CitationPanelReaderInner
      key={`${activeCitation.source_id}__${activeCitation.chunk_id}`}
    />
  );
}

function CitationPanelReaderInner() {
  const { activeCitation } = useAppState();
  const dispatch = useAppDispatch();
  const citation = activeCitation!;

  const handleClose = () =>
    dispatch({ type: "SET_ACTIVE_CITATION", citation: null });

  const [content, setContent] = useState<SourceContentResponse | null>(null);
  const [chunkDetail, setChunkDetail] = useState<ChunkDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const promises: Promise<void>[] = [];

    promises.push(
      sourceApi
        .getContent(citation.source_id)
        .then((data) => { if (!cancelled) setContent(data); })
        .catch((err) => {
          if (!cancelled)
            setError(err instanceof Error ? err.message : "Failed to load content");
        })
    );

    if (citation.chunk_id) {
      promises.push(
        sourceApi
          .getChunk(citation.source_id, citation.chunk_id)
          .then((data) => { if (!cancelled) setChunkDetail(data); })
          .catch(() => { /* non-critical */ })
      );
    }

    Promise.all(promises).finally(() => {
      if (!cancelled) setIsLoading(false);
    });

    return () => { cancelled = true; };
  }, [citation.source_id, citation.chunk_id]);

  const format: "pdf" | "markdown" | "text" =
    chunkDetail?.format ?? content?.format ?? "text";

  const resolvedPageNumber = citation.page ?? chunkDetail?.page_number ?? undefined;
  const resolvedDisplayPage = citation.page != null
    ? String(citation.page)
    : (chunkDetail?.display_page ?? chunkDetail?.page_number ?? undefined);

  const exactHighlightText = citation.highlight_text?.trim();
  const exactHighlightScope = citation.highlight_scope_text?.trim();
  const fallbackChunkText = chunkDetail?.parent_text ?? chunkDetail?.chunk_text ?? citation.chunk_text;
  const fallbackScrollText = chunkDetail?.parent_text
    ? (chunkDetail.chunk_text ?? citation.chunk_text)
    : undefined;

  const highlightPayload: HighlightPayload = exactHighlightText
    ? {
      page_number: resolvedPageNumber,
      header_path: chunkDetail?.header_path ?? citation.header_path ?? undefined,
      chunk_text: exactHighlightText,
      scope_text: exactHighlightScope ?? chunkDetail?.chunk_text ?? chunkDetail?.parent_text ?? citation.chunk_text,
    }
    : {
      page_number: resolvedPageNumber,
      header_path: chunkDetail?.header_path ?? citation.header_path ?? undefined,
      // Highlight the full parent chunk (expanded context); scroll to the child
      // chunk position within it so the view lands on the cited content.
      chunk_text: fallbackChunkText,
      scroll_to_text: fallbackScrollText,
    };

  const sourcePreviewText = chunkDetail?.chunk_text ?? citation.chunk_text;

  const hasHighlightData = !!(
    highlightPayload.chunk_text ||
    highlightPayload.header_path ||
    highlightPayload.page_number
  );

  const formatBadgeColors: Record<string, string> = {
    pdf: "bg-white/10 text-gray-100 border-white/20",
    markdown: "bg-white/10 text-gray-100 border-white/20",
    text: "bg-gray-800 text-gray-300 border-gray-700",
  };

  return (
    <div className="flex flex-col h-full">
      {/* Panel reader header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b shrink-0" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
        <button
          onClick={handleClose}
          className="p-1 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors shrink-0"
          aria-label="Back to sources"
          title="Back to sources"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <div className="flex items-center gap-1.5 min-w-0 flex-1">
          <span className="text-xs font-medium text-foreground truncate">{citation.source_id}</span>
          <span className={`shrink-0 px-1.5 py-0.5 text-[10px] rounded border ${formatBadgeColors[format] ?? formatBadgeColors.text}`}>
            {format.toUpperCase()}
          </span>
          {resolvedPageNumber != null && (
            <span className="shrink-0 px-1.5 py-0.5 text-[10px] rounded border bg-gray-800 text-gray-400 border-gray-700">
              p.{resolvedDisplayPage}
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-3 py-3 text-sm">
        {isLoading && (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-3.5 h-3.5 border-2 border-gray-600 border-t-white rounded-full animate-spin" />
              <span className="text-xs">Loading…</span>
            </div>
          </div>
        )}
        {error && (
          <p className="text-xs text-red-400 px-1">{error}</p>
        )}
        {content && !isLoading && !error && (
          <>
            <DocumentRenderer
              content={content.content}
              format={format}
              highlight={hasHighlightData ? highlightPayload : undefined}
            />
            {sourcePreviewText && (
              <div className="mt-4 p-2.5 rounded-lg border border-gray-700 bg-gray-800/50">
                <p className="text-[10px] text-gray-400 mb-1 font-medium uppercase tracking-wide">Source excerpt</p>
                <p className="text-xs text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {sourcePreviewText}
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
