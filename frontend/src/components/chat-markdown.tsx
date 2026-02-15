"use client";

import ReactMarkdown from "react-markdown";
import { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { CitationEntry, CitationPayload } from "@/lib/api-client";

interface ChatMarkdownProps {
  content: string;
  className?: string;
  citations?: CitationEntry[];
  onCitationClick?: (payload: CitationPayload) => void;
}

function addCitationLinks(content: string, citations?: CitationEntry[]): string {
  if (!citations || citations.length === 0) return content;

  const withChunkMarkersLinked = content.replace(/\[CHUNK\s+(\d+)\](?!\()/gi, (fullMatch, numberText) => {
    const index = Number.parseInt(numberText, 10) - 1;
    if (!Number.isInteger(index) || index < 0 || index >= citations.length) {
      return fullMatch;
    }
    return `[${numberText}](citation:${numberText})`;
  });

  return withChunkMarkersLinked.replace(/\[(\d+)\](?!\()/g, (fullMatch, numberText) => {
    const index = Number.parseInt(numberText, 10) - 1;
    if (!Number.isInteger(index) || index < 0 || index >= citations.length) {
      return fullMatch;
    }
    return `[${numberText}](citation:${numberText})`;
  });
}

function extractCitationIndex(href?: string): number | null {
  if (!href) return null;

  const decodedHref = (() => {
    try {
      return decodeURIComponent(href);
    } catch {
      return href;
    }
  })();

  const citationMatch =
    decodedHref.match(/(?:^|\/)citation:(\d+)(?:$|[/?#])/i) ??
    decodedHref.match(/citation:(\d+)/i);
  if (!citationMatch) return null;

  const index = Number.parseInt(citationMatch[1], 10) - 1;
  return Number.isInteger(index) && index >= 0 ? index : null;
}

export function ChatMarkdown({
  content,
  className = "",
  citations,
  onCitationClick,
}: ChatMarkdownProps) {
  const markdownContent = addCitationLinks(content, citations);

  return (
    <div className={`text-sm leading-relaxed min-h-[1.5em] break-words ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => {
          if (url.toLowerCase().startsWith("citation:")) return url;
          return defaultUrlTransform(url);
        }}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-current/30 pl-3 italic my-2">{children}</blockquote>
          ),
          pre: ({ children }) => (
            <pre className="bg-black/20 rounded-md p-3 overflow-x-auto my-2">{children}</pre>
          ),
          code: ({ children }) => (
            <code className="font-mono text-[0.9em] bg-black/20 px-1 py-0.5 rounded">{children}</code>
          ),
          a: ({ href, children }) => {
            const index = extractCitationIndex(href);
            if (index !== null) {
              const citation =
                citations && Number.isInteger(index) && index >= 0 && index < citations.length
                  ? citations[index]
                  : undefined;
              const payload: CitationPayload | undefined = citation
                ? {
                    source_id: citation.source_id,
                    chunk_id: citation.chunk_id,
                    page_number: citation.page_number,
                    display_page: citation.display_page,
                    header_path: citation.header_path,
                    chunk_text: citation.chunk_text,
                  }
                : undefined;

              return (
                <button
                  type="button"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (payload && onCitationClick) onCitationClick(payload);
                  }}
                  className="inline-flex items-center justify-center w-4 h-4 text-[10px] font-bold text-blue-400 hover:text-blue-300 bg-blue-500/20 hover:bg-blue-500/30 rounded-full align-super ml-0.5 mr-0.5 cursor-pointer transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                  title={payload ? `View source: ${payload.source_id}` : "Source"}
                  disabled={!payload || !onCitationClick}
                >
                  {children}
                </button>
              );
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:underline"
              >
                {children}
              </a>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-2">
              <table className="w-full border-collapse text-left">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-gray-600 px-2 py-1 font-semibold">{children}</th>
          ),
          td: ({ children }) => <td className="border border-gray-700 px-2 py-1">{children}</td>,
        }}
      >
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
}