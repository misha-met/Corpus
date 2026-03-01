"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";

export interface UploadRequest {
  file: File;
  sourceId: string;
  summarize: boolean;
  /** Optional citation reference string for Harvard/footnote copy feature.
   *  e.g. "Smith, J. et al. (2024) 'Climate Change Review'"
   *  Stored in localStorage keyed by source_id. */
  citationRef?: string;
}

interface IngestModalProps {
  onClose: () => void;
  /** Called when the user confirms upload. Modal closes immediately after. */
  onStartUpload: (reqs: UploadRequest[]) => void;
}

const ALLOWED_EXTENSIONS = [".pdf", ".md", ".markdown"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

function sanitizeSourceId(filename: string): string {
  const stem = filename.replace(/\.[^.]+$/, "");
  return stem
    .replace(/[^\w-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 120) || "uploaded_doc";
}

export function IngestModal({ onClose, onStartUpload }: IngestModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [sourceIds, setSourceIds] = useState<string[]>([]);
  const [citationRefs, setCitationRefs] = useState<string[]>([]);
  const [showCitationRefs, setShowCitationRefs] = useState(false);
  const [summarize, setSummarize] = useState(true);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Focus trap + Escape to close
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    modalRef.current?.focus();
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const validateFile = useCallback((f: File): string | null => {
    const ext = "." + f.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type "${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`;
    }
    if (f.size > MAX_FILE_SIZE) {
      return `File too large (${(f.size / (1024 * 1024)).toFixed(1)}MB). Maximum: ${MAX_FILE_SIZE / (1024 * 1024)}MB.`;
    }
    if (f.size === 0) {
      return "File is empty.";
    }
    return null;
  }, []);

  const handleFileSelect = useCallback(
    (selected: FileList | File[]) => {
      const incoming = Array.from(selected);
      if (incoming.length === 0) return;

      const valid: File[] = [];
      for (const f of incoming) {
        const error = validateFile(f);
        if (error) {
          setValidationError(`${f.name}: ${error}`);
          return;
        }
        valid.push(f);
      }

      setFiles(valid);
      setSourceIds(valid.map((f) => sanitizeSourceId(f.name)));
      setCitationRefs(valid.map(() => ""));
      setValidationError(null);
    },
    [validateFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files;
      if (dropped.length > 0) handleFileSelect(dropped);
    },
    [handleFileSelect]
  );

  const handleUpload = useCallback(() => {
    if (files.length === 0) return;

    const normalizedSourceIds = files.map((file, idx) => {
      const fromInput = sourceIds[idx]?.trim() ?? "";
      return fromInput || sanitizeSourceId(file.name);
    });

    if (normalizedSourceIds.some((sid) => !sid)) {
      setValidationError("Each file must have a non-empty Source ID.");
      return;
    }

    const uniqueIds = new Set(normalizedSourceIds);
    if (uniqueIds.size !== normalizedSourceIds.length) {
      setValidationError("Source IDs must be unique across selected files.");
      return;
    }

    const reqs: UploadRequest[] = files.map((file, idx) => ({
      file,
      sourceId: normalizedSourceIds[idx],
      summarize,
      citationRef: citationRefs[idx]?.trim() || undefined,
    }));

    onStartUpload(reqs);
    onClose();
  }, [files, sourceIds, citationRefs, summarize, onStartUpload, onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={modalRef}
        tabIndex={-1}
        className="rounded-2xl w-full max-w-md mx-4 outline-none backdrop-blur-xl"
        style={{
          background: "rgba(10,10,10,0.92)",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow: "0 24px 64px rgba(0,0,0,0.8), 0 4px 16px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <h2 className="text-sm font-semibold text-[var(--foreground)]">
            Add Source Document
          </h2>
          <button
            onClick={onClose}
            className="p-1 text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-white/5 rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4">
          {/* Drag & drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center gap-2 px-4 py-8 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
              dragOver
                ? "border-white/50 bg-white/8"
                : files.length > 0
                ? "border-white/25 bg-white/4"
                : "hover:border-white/20"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.md,.markdown"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileSelect(e.target.files);
                }
                e.target.value = "";
              }}
            />

            {files.length > 0 ? (
              <>
                <svg className="w-8 h-8 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-sm text-[var(--foreground)] font-medium truncate max-w-full">
                  {files.length === 1 ? files[0].name : `${files.length} files selected`}
                </p>
                <p className="text-xs text-[var(--muted-foreground)]">
                  {files.length === 1
                    ? `${(files[0].size / 1024).toFixed(0)} KB`
                    : `${(files.reduce((total, f) => total + f.size, 0) / (1024 * 1024)).toFixed(1)} MB total`}
                  {" "}&middot; Click to change
                </p>
              </>
            ) : (
              <>
                <svg className="w-8 h-8 text-[var(--muted-foreground)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <p className="text-sm text-[var(--muted-foreground)]">
                  Drop a file here or <span className="text-[var(--foreground)]">browse</span>
                </p>
                <p className="text-xs text-[var(--muted-foreground)]/60">
                  PDF, Markdown &middot; Up to 50 MB
                </p>
              </>
            )}
          </div>

          {/* Source ID input(s) */}
          {files.length <= 1 ? (
            <div>
              <label className="block text-xs text-[var(--muted-foreground)] mb-1.5">
                Source ID
              </label>
              <input
                type="text"
                value={sourceIds[0] ?? ""}
                onChange={(e) =>
                  setSourceIds((prev) => {
                    const next = [...prev];
                    next[0] = e.target.value;
                    return next;
                  })
                }
                placeholder="Auto-generated from filename"
                className="w-full px-3 py-2 rounded-lg text-sm text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:outline-none transition-colors"
                style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)" }}
              />
              <p className="mt-1 text-[11px] text-[var(--muted-foreground)]/60">
                Unique identifier. Letters, numbers, hyphens, underscores.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-[var(--muted-foreground)]">Source IDs</p>
              <div className="max-h-40 overflow-y-auto space-y-2 pr-1">
                {files.map((file, idx) => (
                  <div key={`${file.name}-${file.size}-${idx}`}>
                    <label className="block text-[11px] text-[var(--muted-foreground)] mb-1 truncate" title={file.name}>
                      {file.name}
                    </label>
                    <input
                      type="text"
                      value={sourceIds[idx] ?? ""}
                      onChange={(e) =>
                        setSourceIds((prev) => {
                          const next = [...prev];
                          next[idx] = e.target.value;
                          return next;
                        })
                      }
                      className="w-full px-3 py-1.5 rounded-lg text-sm text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:outline-none"
                      style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)" }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Citation reference — optional, collapsible */}
          {files.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() => setShowCitationRefs((v) => !v)}
                className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors mb-1"
              >
                <svg
                  className={`w-3 h-3 shrink-0 transition-transform ${showCitationRefs ? "rotate-90" : ""}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                Citation reference
                <span className="ml-1 text-[10px] px-1.5 py-px rounded bg-white/8 text-[var(--muted-foreground)]">optional</span>
              </button>

              {showCitationRefs && (
                <div className="space-y-2 pl-1">
                  {/* Format hint */}
                  <div
                    className="px-3 py-2.5 rounded-lg text-[11px] text-[var(--muted-foreground)] leading-relaxed space-y-1"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
                  >
                    <p className="font-semibold text-[var(--foreground)]">Harvard author-date format</p>
                    <p>Enter the full reference as you want it to appear in citations:</p>
                    <code className="block mt-1 font-mono text-[10.5px] bg-black/30 px-2 py-1 rounded text-white/70">
                      Smith, J. et al. (2024) &apos;Title of Work&apos;
                    </code>
                    <p className="mt-1 text-[var(--muted-foreground)]/70">
                      Leave blank to use the filename as the reference.
                      You can also add this later — the reference is stored
                      locally and used whenever you copy citations.
                    </p>
                  </div>

                  {files.length <= 1 ? (
                    <input
                      type="text"
                      value={citationRefs[0] ?? ""}
                      onChange={(e) =>
                        setCitationRefs((prev) => {
                          const next = [...prev];
                          next[0] = e.target.value;
                          return next;
                        })
                      }
                      placeholder="e.g. Smith, J. et al. (2024) 'Climate Change Review'"
                      className="w-full px-3 py-2 rounded-lg text-sm text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:outline-none transition-colors"
                      style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)" }}
                    />
                  ) : (
                    <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                      {files.map((file, idx) => (
                        <div key={`citref-${file.name}-${idx}`}>
                          <label className="block text-[11px] text-[var(--muted-foreground)] mb-1 truncate" title={file.name}>
                            {file.name}
                          </label>
                          <input
                            type="text"
                            value={citationRefs[idx] ?? ""}
                            onChange={(e) =>
                              setCitationRefs((prev) => {
                                const next = [...prev];
                                next[idx] = e.target.value;
                                return next;
                              })
                            }
                            placeholder="e.g. Smith, J. (2024) 'Title'"
                            className="w-full px-3 py-1.5 rounded-lg text-sm text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:outline-none"
                            style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)" }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Summarize checkbox */}
          <label className="flex items-center gap-2.5 cursor-pointer" onClick={() => setSummarize((v) => !v)}>
            <Checkbox
              checked={summarize}
              onChange={setSummarize}
            />
            <span className="text-sm text-[var(--muted-foreground)]">
              Generate summary during ingest
            </span>
          </label>

          {/* Validation error */}
          {validationError && (
            <div className="px-3 py-2 bg-red-900/20 border border-red-800/40 rounded-lg text-sm text-red-400">
              {validationError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4" style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-white/5 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={
              files.length === 0 ||
              sourceIds.some((sid) => !(sid ?? "").trim())
            }
            className="px-4 py-2 bg-white text-black hover:bg-white/90 disabled:bg-[var(--secondary)] disabled:text-[var(--muted-foreground)] text-sm font-medium rounded-lg transition-colors disabled:cursor-not-allowed"
          >
            {files.length > 1 ? `Upload & Ingest ${files.length} Files` : "Upload & Ingest"}
          </button>
        </div>
      </div>
    </div>
  );
}
