"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import {
  sourceApi,
  type IngestResponse,
  type NERDiagnostics,
  type SourceInfo,
} from "@/lib/api-client";
import { IngestModal, type UploadRequest } from "@/components/ingest-modal";
import { CitationPanelReader } from "@/components/citation-viewer-modal";
import { File } from "@/components/assistant-ui/file";
import { useAppState, useAppDispatch } from "@/context/app-context";
import { Checkbox } from "@/components/ui/checkbox";
import { MarkdownRenderer } from "@/components/markdown-renderer";

interface SourcePanelProps {
  selectedSourceIds: string[];
  onSelectedSourceIdsChange: (ids: string[]) => void;
  onCollapse: () => void;
  onSourcesChanged?: () => void;
}

const SOURCE_SELECTION_STORAGE_KEY = "dh-selected-source-ids-v1";

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
  onSourcesChanged,
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
  const [ingestDiagnosticsBySourceId, setIngestDiagnosticsBySourceId] = useState<
    Record<
      string,
      {
        geotagNer: NERDiagnostics | null;
        peopletagNer: NERDiagnostics | null;
      }
    >
  >({});
  const [uploadStatus, setUploadStatus] = useState<
    | null
    | { stage: "uploading"; fileName: string; sourceId: string; queued: number; geotag: boolean; peopletag: boolean }
    | { stage: "error"; fileName: string; message: string }
  >(null);
  const uploadDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const persistedSelectionRef = useRef<string[] | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(SOURCE_SELECTION_STORAGE_KEY);
      if (raw === null) {
        persistedSelectionRef.current = null;
        return;
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        persistedSelectionRef.current = [];
        onSelectedSourceIdsChange([]);
        return;
      }
      const hydrated = parsed
        .map((value) => String(value).trim())
        .filter((value, index, arr) => value.length > 0 && arr.indexOf(value) === index);
      persistedSelectionRef.current = hydrated;
      onSelectedSourceIdsChange(hydrated);
    } catch {
      persistedSelectionRef.current = null;
    }
  }, [onSelectedSourceIdsChange]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(SOURCE_SELECTION_STORAGE_KEY, JSON.stringify(selectedSourceIds));
  }, [selectedSourceIds]);

  const fetchSources = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await sourceApi.listSources();
      const nextIds = data.map((source) => source.source_id);
      const nextIdSet = new Set(nextIds);

      let nextSelection: string[]
      const persistedSelection = persistedSelectionRef.current;
      if (persistedSelection !== null) {
        nextSelection = persistedSelection.filter((id) => nextIdSet.has(id));
        // Apply persisted selection once, then continue using live state.
        persistedSelectionRef.current = null;
      } else {
        const previousIds = sources.map((source) => source.source_id);
        const hadAllSelectedBefore =
          previousIds.length > 0 &&
          previousIds.every((id) => selectedSourceIds.includes(id));

        if (hadAllSelectedBefore) {
          nextSelection = nextIds;
        } else {
          const filteredSelection = selectedSourceIds.filter((id) => nextIdSet.has(id));
          if (filteredSelection.length > 0 || selectedSourceIds.length > 0) {
            nextSelection = filteredSelection;
          } else {
            // First load fallback when no persisted choice exists.
            nextSelection = nextIds;
          }
        }
      }

      setSources(data);
      onSourcesChanged?.();

      const sameSelection =
        nextSelection.length === selectedSourceIds.length &&
        nextSelection.every((id, idx) => selectedSourceIds[idx] === id);
      if (!sameSelection) {
        onSelectedSourceIdsChange(nextSelection);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sources");
    } finally {
      setIsLoading(false);
    }
  }, [onSelectedSourceIdsChange, onSourcesChanged, selectedSourceIds, sources]);

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
      geotag: current.geotag,
      peopletag: current.peopletag,
    });

    (async () => {
      try {
        const ingestResponse: IngestResponse = await sourceApi.uploadDocument(
          current.file,
          current.sourceId,
          current.summarize,
          current.pageOffset,
          current.geotag,
          current.peopletag,
          current.citationRef,
        );
        setIngestDiagnosticsBySourceId((prev) => ({
          ...prev,
          [current.sourceId]: {
            geotagNer: ingestResponse.geotag_ner ?? null,
            peopletagNer: ingestResponse.peopletag_ner ?? null,
          },
        }));
        // Refresh the source list after each successful ingest
        const data = await sourceApi.listSources();
        const nextIds = data.map((item) => item.source_id);
        const nextIdSet = new Set(nextIds);
        const previousIds = sources.map((item) => item.source_id);
        const hadAllSelectedBefore =
          previousIds.length > 0 &&
          previousIds.every((id) => selectedSourceIds.includes(id));

        let nextSelection: string[];
        if (hadAllSelectedBefore) {
          nextSelection = nextIds;
        } else {
          nextSelection = selectedSourceIds.filter((id) => nextIdSet.has(id));
          if (!nextSelection.includes(current.sourceId) && nextIdSet.has(current.sourceId)) {
            nextSelection = [...nextSelection, current.sourceId];
          }
        }

        setSources(data);
        onSourcesChanged?.();
        onSelectedSourceIdsChange(nextSelection);
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
  }, [isUploading, onSelectedSourceIdsChange, onSourcesChanged, selectedSourceIds, sources, uploadQueue]);

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

  function handleSelectAll(checked: boolean) {
    onSelectedSourceIdsChange(checked ? sources.map((s) => s.source_id) : []);
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
      setIngestDiagnosticsBySourceId((prev) => {
        if (!(sourceId in prev)) return prev;
        const next = { ...prev };
        delete next[sourceId];
        return next;
      });
      onSourcesChanged?.();
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

  function handleViewFullText(sourceId: string, e: React.MouseEvent) {
    e.stopPropagation();
    dispatch({ type: "SET_ACTIVE_CITATION", citation: { source_id: sourceId, chunk_id: "", number: 0 } });
  }

  function isNerDegraded(diag: NERDiagnostics | null | undefined): boolean {
    if (!diag) return false;
    return !diag.ner_available || diag.method !== "gliner";
  }

  function summarizeNerDegradation(
    sourceId: string,
  ): { visible: boolean; label: string; detail: string } {
    const diag = ingestDiagnosticsBySourceId[sourceId];
    if (!diag) return { visible: false, label: "", detail: "" };

    const parts: string[] = [];
    if (isNerDegraded(diag.geotagNer)) {
      const method = diag.geotagNer?.method ?? "unknown";
      parts.push(`Places: ${method}`);
    }
    if (isNerDegraded(diag.peopletagNer)) {
      const method = diag.peopletagNer?.method ?? "unknown";
      parts.push(`People: ${method}`);
    }

    const detail = parts.join(" · ");
    return {
      visible: parts.length > 0,
      label: "NER degraded",
      detail,
    };
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
      <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "#1e1e1e" }}>
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

      <div className="flex flex-col flex-1 min-h-0">
          {/* Add sources button */}
          <div className="px-4 pt-3 pb-2">
            <button
              onClick={() => setShowIngestModal(true)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm text-white/80 hover:text-white font-medium transition-all"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.10)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04)",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.07)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.04)"; }}
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
            <div className="px-4 py-2 border-b" style={{ borderColor: "#1e1e1e" }}>
              <div className="flex items-center gap-2.5 group">
                <Checkbox
                  id="source-panel-select-all"
                  checked={allSelected}
                  onChange={handleSelectAll}
                />
                <label
                  htmlFor="source-panel-select-all"
                  className="cursor-pointer text-xs text-muted-foreground group-hover:text-foreground transition-colors"
                >
                  Select all sources
                </label>
              </div>
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
            <div className="flex items-center gap-3 px-3 py-3 bg-white/5 border border-white/15 rounded-xl animate-pulse">
              <svg className="w-5 h-5 text-white/80 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <div className="min-w-0">
                <p className="text-sm font-medium text-gray-100 truncate">
                  Ingesting {uploadStatus.fileName}...
                </p>
                <p className="text-xs text-gray-400 mt-0.5">
                  Chunking &amp; embedding
                  {uploadStatus.geotag || uploadStatus.peopletag
                    ? `, ${[
                      uploadStatus.geotag ? "geotagging" : null,
                      uploadStatus.peopletag ? "people indexing" : null,
                    ].filter(Boolean).join(" + ")}`
                    : ""}
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
            const nerDegradation = summarizeNerDegradation(source.source_id);
            return (
              <div
                key={source.source_id}
                className={`px-3 py-2.5 rounded-lg hover:bg-[#1a1a1a] transition-colors group cursor-pointer ${isHighlighted ? "bg-[#1c1c1c]" : ""
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
                  <File.Root
                    variant="ghost"
                    size="sm"
                    className="p-0 border-0 bg-transparent"
                  >
                    <File.Icon
                      mimeType="application/pdf"
                      className="w-8 h-8 text-white/85 shrink-0"
                    />
                  </File.Root>

                  {/* Title */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-[#e0e0e0] truncate">
                      {source.source_id}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground/70 truncate">
                      {isPending
                        ? source.ingestState === "queued"
                          ? "In Ingest Queue"
                          : "Ingesting..."
                        : sizeLabel ?? "Size unavailable"}
                    </p>
                    {nerDegradation.visible && !isPending && (
                      <span
                        className="mt-1 inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-wide"
                        style={{
                          borderColor: "rgba(245, 158, 11, 0.5)",
                          color: "rgba(253, 224, 71, 0.95)",
                          background: "rgba(120, 53, 15, 0.35)",
                        }}
                        title={nerDegradation.detail}
                      >
                        {nerDegradation.label}
                      </span>
                    )}
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
                  <span onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={isChecked}
                      onChange={() => handleToggleSource(source.source_id)}
                      disabled={isPending}
                    />
                  </span>
                </div>

                {isActive && (
                  <div className="mt-3 pl-11 pr-1 space-y-2">
                    <div
                      className="rounded-xl px-3 py-2.5 max-h-40 overflow-y-auto text-xs text-muted-foreground prose prose-invert prose-xs max-w-none [&_p]:leading-relaxed [&_p]:my-1 [&_li]:my-0 [&_ul]:my-1"
                      style={{
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        boxShadow: "inset 0 1px 4px rgba(0,0,0,0.4)",
                      }}
                    >
                      {source.ingestState === "queued" || source.ingestState === "ingesting"
                        ? <p>{source.ingestState === "queued" ? "In Ingest Queue" : "Ingesting..."}</p>
                        : <MarkdownRenderer content={source.summary?.trim() || "No summary available for this source."} />}
                    </div>
                    {!isPending && (
                      <button
                        onClick={(e) => handleViewFullText(source.source_id, e)}
                        className="px-3 py-1.5 text-xs rounded-lg text-[#d0d0d0] transition-all"
                        style={{
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.10)",
                        }}
                      >
                        View full text
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t text-xs text-[#555555]" style={{ borderColor: "#1e1e1e" }}>
            {displaySources.length} source{displaySources.length !== 1 ? "s" : ""} &middot;{" "}
            {selectedSourceIds.length} selected
          </div>
      </div>

      {/* Ingest modal */}
      {showIngestModal && (
        <IngestModal
          onClose={() => setShowIngestModal(false)}
          onStartUpload={handleStartUpload}
          existingSourceIds={displaySources.map((s) => s.source_id)}
        />
      )}
    </div>
  );
}
