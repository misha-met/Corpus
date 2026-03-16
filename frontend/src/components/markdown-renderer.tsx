"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { findAndHighlight } from "@/lib/text-highlighter";

interface HighlightPayload {
  page_number?: number | null;
  header_path?: string;
  chunk_text?: string;
  scroll_to_text?: string;
  scope_text?: string;
}

interface MarkdownRendererProps {
  content: string;
  highlight?: HighlightPayload;
}

/** Slugify a heading string for use as an element id. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .trim();
}

/**
 * Renders Markdown content with optional scroll-to-section and text
 * highlighting for citation viewer.
 *
 * - Heading IDs are generated from heading text (slug) so we can
 *   scroll to sections identified by `header_path`.
 * - After mount, if `highlight.chunk_text` is set, we search for it
 *   in the rendered DOM and wrap the match in `<mark>`.
 * - Falls back to scrolling to the section heading if text match
 *   fails but `header_path` is available.
 */
export function MarkdownRenderer({ content, highlight }: MarkdownRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Give the DOM a tick to render before searching
    const timer = setTimeout(() => {
      let scrollTarget: HTMLElement | null = null;

      // Try to find and highlight the chunk text in the rendered DOM.
      const searchText = highlight?.chunk_text;
      if (searchText) {
        const mark = findAndHighlight(el, searchText, highlight?.scroll_to_text, highlight?.scope_text);
        if (mark) {
          scrollTarget = mark;
        }
      }

      // 2. Fallback: scroll to section heading from header_path
      if (!scrollTarget && highlight?.header_path) {
        const segments = highlight.header_path.split(">").map((s) => s.trim());
        // Try last segment first (most specific), then work upward
        for (let i = segments.length - 1; i >= 0; i--) {
          const slug = slugify(segments[i]);
          if (!slug) continue;
          const heading = el.querySelector(`[id="${slug}"]`);
          if (heading) {
            scrollTarget = heading as HTMLElement;
            break;
          }
        }
      }

      if (scrollTarget) {
        scrollTarget.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);

    return () => clearTimeout(timer);
  }, [content, highlight]);

  return (
    <div
      ref={containerRef}
      className="markdown-content prose prose-invert prose-sm max-w-none
        prose-headings:text-gray-200 prose-p:text-gray-300 prose-li:text-gray-300
        prose-a:text-blue-400 prose-strong:text-gray-200 prose-code:text-gray-300
        prose-pre:bg-gray-800 prose-pre:border prose-pre:border-gray-700"
    >
      <ReactMarkdown
        components={{
          ...Object.fromEntries(
            (["h1", "h2", "h3", "h4", "h5", "h6"] as const).map((Tag) => [
              Tag,
              ({ children, ...props }: React.ComponentProps<typeof Tag>) => (
                <Tag id={slugify(String(children))} {...props}>
                  {children}
                </Tag>
              ),
            ]),
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
