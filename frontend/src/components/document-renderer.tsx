"use client";

import { useEffect, useRef } from "react";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { findAndHighlight } from "@/lib/text-highlighter";

export interface HighlightPayload {
  page_number?: number | null;
  header_path?: string;
  chunk_text?: string;
  /** Corrected highlight passage from parent chunk (post-hoc verification). */
  highlight_text?: string;
}

interface DocumentRendererProps {
  content: string;
  format: "pdf" | "markdown" | "text";
  highlight?: HighlightPayload;
}

/**
 * Plain text renderer with optional text highlighting.
 * Used for PDF-extracted text and raw text files.
 */
function PlainTextRenderer({
  content,
  highlight,
}: {
  content: string;
  highlight?: HighlightPayload;
}) {
  const containerRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const searchText = highlight?.chunk_text || highlight?.highlight_text;
    const needleSource = highlight?.chunk_text ? "chunk_text" : "highlight_text";
    if (!searchText) return;

    const timer = setTimeout(() => {
      // Try full chunk_text first, then fall back to highlight_text.
      let mark = findAndHighlight(el, searchText);
      if (!mark && highlight?.chunk_text && highlight?.highlight_text) {
        mark = findAndHighlight(el, highlight.highlight_text);
      }
      if (mark) {
        mark.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);

    return () => clearTimeout(timer);
  }, [content, highlight]);

  return (
    <pre
      ref={containerRef}
      className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-gray-300 bg-transparent p-0 m-0"
    >
      {content}
    </pre>
  );
}

/**
 * Routes to the appropriate renderer based on document format.
 *
 * - `markdown`: Uses MarkdownRenderer (react-markdown with heading IDs
 *   and fuzzy text highlight).
 * - `pdf` / `text`: Uses PlainTextRenderer (extracted text with fuzzy
 *   text highlight).  PDF canvas rendering via react-pdf can be added
 *   later when a `/file` endpoint is available.
 */
export function DocumentRenderer({
  content,
  format,
  highlight,
}: DocumentRendererProps) {
  if (format === "markdown") {
    return <MarkdownRenderer content={content} highlight={highlight} />;
  }

  // PDF extracted text and plain text both use the plain renderer for now.
  // When a file-serving endpoint is added, PDF format can switch to
  // react-pdf canvas rendering.
  return <PlainTextRenderer content={content} highlight={highlight} />;
}
