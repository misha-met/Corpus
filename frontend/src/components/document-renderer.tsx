"use client";

import { useEffect, useRef } from "react";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { findAndHighlight } from "@/lib/text-highlighter";

export interface HighlightPayload {
  page_number?: number | null;
  header_path?: string;
  /** Text to highlight (typically the full parent chunk). */
  chunk_text?: string;
  /** Text to scroll to within the highlighted region (typically the child chunk). */
  scroll_to_text?: string;
  /** Optional scope text to disambiguate repeated chunk_text matches. */
  scope_text?: string;
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

    const searchText = highlight?.chunk_text;
    if (!searchText) return;

    const timer = setTimeout(() => {
      const mark = findAndHighlight(el, searchText, highlight?.scroll_to_text, highlight?.scope_text);
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
 *   and text highlight).
 * - `pdf` / `text`: Uses PlainTextRenderer (extracted text with
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
