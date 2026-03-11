"use client";

import ReactMarkdown from "react-markdown";
import { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/event-parser";
import { useAppDispatch } from "@/context/app-context";

interface ChatMarkdownProps {
  content: string;
  className?: string;
  citations?: Citation[];
}

function addCitationLinks(content: string, citations?: Citation[]): string {
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
  citations = [],
}: ChatMarkdownProps) {
  const dispatch = useAppDispatch();
  const markdownContent = addCitationLinks(content, citations);

  return (
    <div className={`text-base leading-[1.65] min-h-[1.5em] wrap-break-word ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => {
          if (url.toLowerCase().startsWith("citation:")) return url;
          return defaultUrlTransform(url);
        }}
        components={{
          p: ({ children }) => <p className="mb-[1em] last:mb-0">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-white/40 pl-3 italic my-2 text-gray-200">{children}</blockquote>
          ),
          pre: ({ children }) => (
            <pre className="bg-zinc-900 rounded-md p-3 overflow-x-auto my-2 text-sm">{children}</pre>
          ),
          code: ({ children }) => (
            <code className="font-mono text-[0.9em] bg-zinc-900 px-1 py-0.5 rounded">{children}</code>
          ),
          a: ({ href, children }) => {
            const index = extractCitationIndex(href);
            if (index !== null) {
              // Look up citation by 1-based number (index is already 0-based, so add 1)
              const citation: Citation | undefined = citations.find(
                (c) => c.number === index + 1,
              );

              return (
                <span className="inline-flex items-center mx-0.5 align-middle">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      if (citation) {
                        dispatch({ type: "SET_ACTIVE_CITATION", citation });
                      }
                    }}
                    className="inline-flex items-center justify-center min-w-4.5 h-4.5 px-1 text-[10px] font-semibold text-black rounded-full cursor-pointer transition-colors disabled:opacity-60 disabled:cursor-not-allowed bg-white/90 border border-white/80 hover:bg-white"
                    title={citation ? `View source: ${citation.source_id}` : "Source (no metadata)"}
                    disabled={!citation}
                  >
                    {children}
                  </button>
                </span>
              );
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground hover:text-white underline underline-offset-2 decoration-white/40"
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