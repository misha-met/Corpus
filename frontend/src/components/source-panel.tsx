"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { sourceApi, type SourceInfo } from "@/lib/api-client";
import { IngestModal, type UploadRequest } from "@/components/ingest-modal";
import { CitationPanelReader } from "@/components/citation-viewer-modal";
import { useAppState, useAppDispatch } from "@/context/app-context";

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
  type IngestState = "ingesting" | "queued";
  type DisplaySource = SourceInfo & { ingestState?: IngestState };

  const { activeCitation } = useAppState();
  const dispatch = useAppDispatch();

  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showIngestModal, setShowIngestModal] = useState(false);
  const [highlightSourceId, setHighlightSourceId] = useState<string | null>(null);
  const [activeSourceId, setActiveSourceId] = useState<string | null>(null);
  const [uploadQueue, setUploadQueue] = useState<UploadRequest[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [sourceContent, setSourceContent] = useState<Record<string, string>>({});
  const [loadingContentId, setLoadingContentId] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<
    | null
    | { stage: "uploading"; fileName: string; sourceId: string; queued: number }
    | { stage: "error"; fileName: string; message: string }
  >(null);
  const uploadDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  function formatBytes(bytes?: number | null): string | null {
    if (!bytes || bytes <= 0) return null;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /** Queue uploads; processing runs sequentially in an effect. */
  const handleStartUpload = useCallback(
    (reqs: UploadRequest[]) => {
      setShowIngestModal(false);
      if (reqs.length === 0) return;
      setUploadQueue((prev) => [...prev, ...reqs]);
    },
    []
  );

  useEffect(() => {
    if (isUploading || uploadQueue.length === 0) return;

    const current = uploadQueue[0];
    const remaining = uploadQueue.length - 1;
    setUploadQueue((prev) => prev.slice(1));
    setIsUploading(true);
    setUploadStatus({
      stage: "uploading",
      fileName: current.file.name,
      sourceId: current.sourceId,
      queued: remaining,
    });

    (async () => {
      try {
        await sourceApi.uploadDocument(current.file, current.sourceId, current.summarize);
        // Refresh the source list after each successful ingest
        const data = await sourceApi.listSources();
        setSources(data);
        onSelectedSourceIdsChange(data.map((s) => s.source_id));
        // Briefly highlight + open the new source
        setHighlightSourceId(current.sourceId);
        setActiveSourceId(current.sourceId);
        if (uploadDismissTimer.current) clearTimeout(uploadDismissTimer.current);
        uploadDismissTimer.current = setTimeout(() => setHighlightSourceId(null), 3000);
      } catch (err) {
        setUploadStatus({
          stage: "error",
          fileName: current.file.name,
          message: err instanceof Error ? err.message : "Upload failed",
        });
      } finally {
        setIsUploading(false);
      }
    })();
  }, [isUploading, onSelectedSourceIdsChange, uploadQueue]);

  useEffect(() => {
    if (!isUploading && uploadQueue.length === 0 && uploadStatus?.stage === "uploading") {
      setUploadStatus(null);
    }
  }, [isUploading, uploadQueue.length, uploadStatus]);

  const pendingStateById = useMemo(() => {
    const map = new Map<string, IngestState>();
    if (uploadStatus?.stage === "uploading") {
      map.set(uploadStatus.sourceId, "ingesting");
    }
    for (const queued of uploadQueue) {
      if (!map.has(queued.sourceId)) {
        map.set(queued.sourceId, "queued");
      }
    }
    return map;
  }, [uploadQueue, uploadStatus]);

  const displaySources = useMemo<DisplaySource[]>(() => {
    const list: DisplaySource[] = sources.map((s) => ({
      ...s,
      ingestState: pendingStateById.get(s.source_id),
    }));

    const known = new Set(list.map((s) => s.source_id));
    for (const [sourceId, ingestState] of pendingStateById.entries()) {
      if (known.has(sourceId)) continue;
      list.push({
        source_id: sourceId,
        summary: ingestState === "queued" ? "In Ingest Queue" : "Ingesting...",
        source_path: null,
        snapshot_path: null,
        source_size_bytes: null,
        content_size_bytes: null,
        ingestState,
      });
    }
    return list;
  }, [pendingStateById, sources]);

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
      setSourceContent((prev) => {
        const next = { ...prev };
        delete next[sourceId];
        return next;
      });
      if (activeSourceId === sourceId) {
        setActiveSourceId(null);
      }
      onSelectedSourceIdsChange(
        selectedSourceIds.filter((id) => id !== sourceId)
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to delete source"
      );
    }
  }

  async function handleViewFullText(sourceId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (sourceContent[sourceId]) {
      setSourceContent((prev) => {
        const next = { ...prev };
        delete next[sourceId];
        return next;
      });
      return;
    }
    setContentError(null);
    setLoadingContentId(sourceId);
    try {
      const content = await sourceApi.getContent(sourceId);
      setSourceContent((prev) => ({ ...prev, [sourceId]: content.content }));
    } catch (err) {
      setContentError(err instanceof Error ? err.message : "Failed to load content");
    } finally {
      setLoadingContentId(null);
    }
  }

  /* ── Citation reader mode — slides in over the file list ── */
  if (activeCitation !== null) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <CitationPanelReader />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
        <h2 className="text-sm font-semibold text-foreground tracking-wide">
          Sources
        </h2>
        <div className="flex items-center gap-1">
          <button
            onClick={fetchSources}
            disabled={isLoading}
            className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors disabled:opacity-50"
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
            className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors"
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
          onClick={() => setShowIngestModal(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 rounded-xl text-sm text-muted-foreground hover:text-foreground font-medium transition-colors"
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
      {displaySources.length > 0 && (
        <div className="px-4 py-2 border-b" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
          <label className="flex items-center gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={handleSelectAll}
              className="w-4 h-4 rounded border-white/20 bg-white/5 accent-indigo-500 cursor-pointer"
            />
            <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
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

        {isLoading && displaySources.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-gray-500">
            Loading sources...
          </div>
        )}

        {!isLoading && displaySources.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-gray-500">
            <p>No sources ingested yet.</p>
            <p className="mt-2 text-xs text-gray-600">
              Click &ldquo;Add sources&rdquo; to upload a document
            </p>
          </div>
        )}

        <div className="px-3 py-1 space-y-1">
          {/* Inline upload status */}
          {uploadStatus?.stage === "uploading" && (
            <div className="flex items-center gap-3 px-3 py-3 bg-blue-900/20 border border-blue-800/30 rounded-xl animate-pulse">
              <svg className="w-5 h-5 text-blue-400 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <div className="min-w-0">
                <p className="text-sm font-medium text-blue-300 truncate">
                  Ingesting {uploadStatus.fileName}...
                </p>
                <p className="text-xs text-blue-400/70 mt-0.5">
                  Chunking, embedding &amp; summarizing
                  {uploadStatus.queued > 0 ? ` · ${uploadStatus.queued} queued` : ""}
                </p>
              </div>
            </div>
          )}

          {uploadStatus?.stage === "error" && (
            <div className="flex items-start gap-3 px-3 py-3 bg-red-900/20 border border-red-800/40 rounded-xl">
              <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-red-300 truncate">
                  Failed to ingest {uploadStatus.fileName}
                </p>
                <p className="text-xs text-red-400/80 mt-0.5">
                  {uploadStatus.message}
                </p>
              </div>
              <button
                onClick={() => setUploadStatus(null)}
                className="shrink-0 p-0.5 text-red-400/60 hover:text-red-300 transition-colors"
                title="Dismiss"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
          {displaySources.map((source) => {
            const isChecked = selectedSourceIds.includes(source.source_id);
            const isHighlighted = highlightSourceId === source.source_id;
            const isActive = activeSourceId === source.source_id;
            const isPending = source.ingestState === "ingesting" || source.ingestState === "queued";
            const sizeLabel = formatBytes(source.content_size_bytes ?? source.source_size_bytes);
            const hasContent = Boolean(sourceContent[source.source_id]);
            return (
              <div
                key={source.source_id}
                className={`px-3 py-2.5 rounded-lg hover:bg-white/5 transition-colors group cursor-pointer ${
                  isHighlighted ? "ring-2 ring-blue-500/60 bg-blue-900/20" : ""
                }`}
                onClick={() =>
                  setActiveSourceId((prev) =>
                    prev === source.source_id ? null : source.source_id
                  )
                }
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setActiveSourceId((prev) =>
                      prev === source.source_id ? null : source.source_id
                    );
                  }
                }}
              >
                <div className="flex items-center gap-3">
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
                    <p className="text-sm font-medium text-foreground truncate">
                      {source.source_id}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground/70 truncate">
                      {isPending
                        ? source.ingestState === "queued"
                          ? "In Ingest Queue"
                          : "Ingesting..."
                        : sizeLabel ?? "Size unavailable"}
                    </p>
                  </div>

                  {/* Delete (hover) */}
                  {!isPending && (
                    <button
                      onClick={(e) => handleDelete(source.source_id, e)}
                      className="shrink-0 p-1 text-muted-foreground/40 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
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
                  )}

                  {/* Checkbox */}
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleToggleSource(source.source_id)}
                    onClick={(e) => e.stopPropagation()}
                    disabled={isPending}
                    className="w-4 h-4 rounded border-white/20 bg-white/5 accent-indigo-500 cursor-pointer shrink-0 disabled:opacity-50"
                  />
                </div>

                {isActive && (
                  <div className="mt-3 pl-11 pr-1 space-y-2">
                    <div className="text-xs text-gray-300 whitespace-pre-wrap leading-relaxed max-h-28 overflow-y-auto rounded-md bg-gray-900/60 border border-gray-800 px-2 py-2">
                      {source.ingestState === "queued"
                        ? "In Ingest Queue"
                        : source.ingestState === "ingesting"
                        ? "Ingesting..."
                        : source.summary?.trim() || "No summary available for this source."}
                    </div>
                    {!isPending && (
                      <div className="flex items-center justify-between gap-2">
                        <button
                          onClick={(e) => handleViewFullText(source.source_id, e)}
                          disabled={loadingContentId === source.source_id}
                          className="px-2.5 py-1.5 text-xs rounded-md bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-200 disabled:opacity-60 disabled:cursor-not-allowed"
                        >
                          {hasContent
                            ? "Hide full text"
                            : loadingContentId === source.source_id
                            ? "Loading..."
                            : "View full text"}
                        </button>
                      </div>
                    )}

                    {contentError && loadingContentId !== source.source_id && (
                      <p className="text-xs text-red-400">{contentError}</p>
                    )}

                    {!isPending && hasContent && (
                      <pre className="text-xs text-gray-300 whitespace-pre-wrap max-h-48 overflow-y-auto rounded-md bg-gray-950 border border-gray-800 px-2 py-2">
                        {sourceContent[source.source_id]}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t text-xs text-muted-foreground/50" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
        {displaySources.length} source{displaySources.length !== 1 ? "s" : ""} &middot;{" "}
        {selectedSourceIds.length} selected
      </div>

      {/* Ingest modal */}
      {showIngestModal && (
        <IngestModal
          onClose={() => setShowIngestModal(false)}
          onStartUpload={handleStartUpload}
        />
      )}
    </div>
  );
}
