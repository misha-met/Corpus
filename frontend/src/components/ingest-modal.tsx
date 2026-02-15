"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UploadRequest {
  file: File;
  sourceId: string;
  summarize: boolean;
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
    }));

    onStartUpload(reqs);
    onClose();
  }, [files, sourceIds, summarize, onStartUpload, onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={modalRef}
        tabIndex={-1}
        className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-md mx-4 outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <h2 className="text-base font-semibold text-gray-100">
            Add Source Document
          </h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4">
          {/* Drag & drop zone / file picker */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center gap-2 px-4 py-8 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
              dragOver
                ? "border-blue-500 bg-blue-500/10"
                : files.length > 0
                ? "border-green-600/50 bg-green-900/10"
                : "border-gray-700 hover:border-gray-600 bg-gray-800/30"
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
                // Reset so same file can be re-selected
                e.target.value = "";
              }}
            />

            {files.length > 0 ? (
              <>
                <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-sm text-gray-200 font-medium truncate max-w-full">
                  {files.length === 1 ? files[0].name : `${files.length} files selected`}
                </p>
                <p className="text-xs text-gray-500">
                  {files.length === 1
                    ? `${(files[0].size / 1024).toFixed(0)} KB`
                    : `${(files.reduce((total, f) => total + f.size, 0) / (1024 * 1024)).toFixed(1)} MB total`}
                  {" "}&middot; Click to change
                </p>
              </>
            ) : (
              <>
                <svg className="w-8 h-8 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <p className="text-sm text-gray-400">
                  Drop a file here or <span className="text-blue-400">browse</span>
                </p>
                <p className="text-xs text-gray-600">
                  PDF, Markdown &middot; Up to 50MB
                </p>
              </>
            )}
          </div>

          {/* Source ID input(s) */}
          {files.length <= 1 ? (
            <div>
              <label className="block text-xs text-gray-400 mb-1.5">
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
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              />
              <p className="mt-1 text-[11px] text-gray-600">
                Unique identifier. Letters, numbers, hyphens, underscores.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-gray-400">Source IDs</p>
              <div className="max-h-40 overflow-y-auto space-y-2 pr-1">
                {files.map((file, idx) => (
                  <div key={`${file.name}-${file.size}-${idx}`}>
                    <label className="block text-[11px] text-gray-500 mb-1 truncate" title={file.name}>
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
                      className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Summarize checkbox */}
          <label className="flex items-center gap-2.5 cursor-pointer">
            <input
              type="checkbox"
              checked={summarize}
              onChange={(e) => setSummarize(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 focus:ring-1 cursor-pointer"
            />
            <span className="text-sm text-gray-300">
              Generate summary during ingest
            </span>
          </label>

          {/* Validation error */}
          {validationError && (
            <div className="px-3 py-2 bg-red-900/30 border border-red-800/50 rounded-lg text-sm text-red-300">
              {validationError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={
              files.length === 0 ||
              sourceIds.some((sid) => !(sid ?? "").trim())
            }
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors disabled:cursor-not-allowed"
          >
            {files.length > 1 ? `Upload & Ingest ${files.length} Files` : "Upload & Ingest"}
          </button>
        </div>
      </div>
    </div>
  );
}
