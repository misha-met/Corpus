"use client";

import ReactMarkdown from "react-markdown";
import { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/event-parser";
import { useAppState, useAppDispatch } from "@/context/app-context";

interface ChatMarkdownProps {
  content: string;
  className?: string;
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
}: ChatMarkdownProps) {
  const { citations } = useAppState();
  const dispatch = useAppDispatch();
  const markdownContent = addCitationLinks(content, citations);

  return (
    <div className={`text-base leading-[1.65] min-h-[1.5em] break-words ${className}`}>
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
              // Look up citation by 1-based number (index is already 0-based, so add 1)
              const citation: Citation | undefined = citations.find(
                (c) => c.number === index + 1,
              );

              return (
                <sup className="inline-flex items-center ml-0.5 mr-0.5">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      if (citation) {
                        dispatch({ type: "SET_ACTIVE_CITATION", citation });
                      }
                    }}
                    className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] font-semibold text-indigo-300 hover:text-indigo-100 rounded-full cursor-pointer transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                    style={{
                      background: "rgba(99,102,241,0.15)",
                      border: "1px solid rgba(99,102,241,0.35)",
                      lineHeight: 1,
                    }}
                    title={citation ? `View source: ${citation.source_id}` : "Source (no metadata)"}
                    disabled={!citation}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(99,102,241,0.28)"; }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(99,102,241,0.15)"; }}
                  >
                    {children}
                  </button>
                </sup>
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