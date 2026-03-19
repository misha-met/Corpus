"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useAuiState } from "@assistant-ui/react";
import { useAppState, useAppDispatch } from "@/context/app-context";
import { groupCitations } from "@/lib/group-citations";
import {
    extractInlineCitationMarkers,
    formatFootnotesWithText,
    formatHarvardBibliography,
} from "@/lib/format-citations";
import { sourceApi } from "@/lib/api-client";
import { ChevronDownIcon, ChevronRightIcon, Copy, CheckCheck } from "lucide-react";

export function MessageReferences() {
    const messageId = useAuiState((s) => s.message.id);
    const isRunning = useAuiState((s) => s.message.status?.type === "running");
    // Extract message text to determine which citation numbers were actually used
    const messageParts = useAuiState((s) => s.message.content);
    const { citationsByMessage } = useAppState();
    const dispatch = useAppDispatch();
    const citations = useMemo(
        () => citationsByMessage[messageId] ?? [],
        [citationsByMessage, messageId]
    );

    const answerText = useMemo(() => {
        return (messageParts ?? [])
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .filter((p: any) => p.type === "text")
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .map((p: any) => p.text ?? "")
            .join("");
    }, [messageParts]);

    // Build a set of citation numbers that appear in the answer text (e.g. [1], [2])
    const citedNumbers = useMemo(() => {
        const nums = new Set<number>();
        for (const marker of extractInlineCitationMarkers(answerText)) {
            nums.add(marker.number);
        }
        return nums;
    }, [answerText]);

    const markerPageByNumber = useMemo(() => {
        const pageByNumber = new Map<number, number>();
        for (const marker of extractInlineCitationMarkers(answerText)) {
            if (marker.page != null && !pageByNumber.has(marker.number)) {
                pageByNumber.set(marker.number, marker.page);
            }
        }
        return pageByNumber;
    }, [answerText]);

    const citationsWithMarkerPages = useMemo(
        () => citations.map((citation) => {
            const markerPage = markerPageByNumber.get(citation.number);
            return markerPage != null ? { ...citation, page: markerPage } : citation;
        }),
        [citations, markerPageByNumber]
    );

    const [isDrawerOpen, setIsDrawerOpen] = useState(false);
    const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
    const [chunkCache, setChunkCache] = useState<Record<string, string>>({});
    const [loadingSources, setLoadingSources] = useState<Set<string>>(new Set());
    const [citationReferenceBySource, setCitationReferenceBySource] = useState<Map<string, string>>(new Map());

    useEffect(() => {
        let cancelled = false;
        sourceApi
            .listSources()
            .then((sources) => {
                if (cancelled) return;
                const map = new Map<string, string>();
                for (const source of sources) {
                    const ref = source.citation_reference?.trim();
                    if (ref) {
                        map.set(source.source_id, ref);
                    }
                }
                setCitationReferenceBySource(map);
            })
            .catch(() => {
            });
        return () => {
            cancelled = true;
        };
    }, []);

    const grouped = useMemo(() => groupCitations(citationsWithMarkerPages), [citationsWithMarkerPages]);

    const toggleDrawer = () => setIsDrawerOpen((prev) => !prev);

    const toggleSource = useCallback(
        async (sourceId: string, chunkIds: string[]) => {
            const isCurrentlyExpanded = expandedSources.has(sourceId);

            setExpandedSources((prev) => {
                const next = new Set(prev);
                if (isCurrentlyExpanded) {
                    next.delete(sourceId);
                } else {
                    next.add(sourceId);
                }
                return next;
            });

            if (isCurrentlyExpanded) return;

            // Fetch missing chunks
            const missingChunkIds = chunkIds.filter((id) => !chunkCache[id]);
            if (missingChunkIds.length > 0) {
                setLoadingSources((prev) => new Set(prev).add(sourceId));
                try {
                    const resp = await sourceApi.getChunks(sourceId, missingChunkIds);
                    setChunkCache((prev) => {
                        const next = { ...prev };
                        for (const chunk of resp.chunks) {
                            next[chunk.chunk_id] = chunk.chunk_text;
                        }
                        return next;
                    });
                } catch (error) {
                    console.error("Failed to fetch chunks for overview:", error);
                } finally {
                    setLoadingSources((prev) => {
                        const next = new Set(prev);
                        next.delete(sourceId);
                        return next;
                    });
                }
            }
        },
        [chunkCache, expandedSources]
    );

    // Only the green (actually-cited) citations, in citation-number order
    const citedCitations = useMemo(
        () => citationsWithMarkerPages.filter((c) => citedNumbers.size === 0 || citedNumbers.has(c.number)),
        [citationsWithMarkerPages, citedNumbers]
    );

    const totalChunks = citations.length;
    const totalSources = grouped.length;

    // Copy-to-clipboard state: null | "footnote" | "harvard"
    const [copied, setCopied] = useState<"footnote" | "harvard" | null>(null);

    function handleCopy(style: "footnote" | "harvard") {
        let text: string;
        if (style === "footnote") {
            text = formatFootnotesWithText(answerText, citedCitations, citationReferenceBySource);
        } else {
            text = formatHarvardBibliography(citedCitations, citationReferenceBySource);
        }
        navigator.clipboard.writeText(text).then(() => {
            setCopied(style);
            setTimeout(() => setCopied(null), 2000);
        });
    }

    if (grouped.length === 0 || isRunning) {
        return null;
    }

    return (
        <div className="mt-2 mb-1 w-full flex flex-col items-start border-t border-white/10 pt-2 pb-1 text-sm transition-all duration-200">
            <div className="flex items-center gap-2 w-full">
                <button
                    onClick={toggleDrawer}
                    className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground outline-none group flex-1 min-w-0 transition-colors"
                >
                    {isDrawerOpen ? (
                        <ChevronDownIcon className="size-4 shrink-0 transition-transform" />
                    ) : (
                        <ChevronRightIcon className="size-4 shrink-0 transition-transform" />
                    )}
                    <span className="font-medium truncate">
                        References ({totalSources} source{totalSources !== 1 ? "s" : ""},{" "}
                        {totalChunks} chunk{totalChunks !== 1 ? "s" : ""})
                    </span>
                </button>

                {/* Copy buttons — only shown when there are cited passages */}
                {citedCitations.length > 0 && (
                    <div className="flex items-center gap-1 shrink-0">
                        <button
                            type="button"
                            onClick={() => handleCopy("footnote")}
                            title="Copy as academic footnotes"
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors
                                text-muted-foreground border-white/10 hover:text-foreground hover:border-white/30"
                        >
                            {copied === "footnote" ? (
                                <CheckCheck className="size-3 text-green-400" />
                            ) : (
                                <Copy className="size-3" />
                            )}
                            Footnotes
                        </button>
                        <button
                            type="button"
                            onClick={() => handleCopy("harvard")}
                            title="Copy as Harvard bibliography"
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors
                                text-muted-foreground border-white/10 hover:text-foreground hover:border-white/30"
                        >
                            {copied === "harvard" ? (
                                <CheckCheck className="size-3 text-green-400" />
                            ) : (
                                <Copy className="size-3" />
                            )}
                            Harvard
                        </button>
                    </div>
                )}
            </div>

            {isDrawerOpen && (
                <div className="w-full mt-3 flex flex-col gap-3 pl-2 sm:pl-4">
                    {grouped.map((group) => {
                        const isExpanded = expandedSources.has(group.sourceId);
                        const isLoading = loadingSources.has(group.sourceId);
                        const chunkIds = group.citations.map((c) => c.chunkId);

                        return (
                            <div key={group.sourceId} className="flex flex-col text-sm text-foreground/90 w-full mb-1">
                                <div className="flex flex-col mb-1 border-l-2 border-[#333] pl-3 py-0.5">
                                    <div className="font-semibold text-foreground break-words line-clamp-2">
                                        {citationReferenceBySource.get(group.sourceId) ?? group.displayName}
                                    </div>
                                    <div className="flex items-center gap-1.5 flex-wrap mt-1">
                                        {group.citations.map((cit, idx) => {
                                            const isUsed = citedNumbers.size === 0 || citedNumbers.has(cit.index);
                                            return (
                                            <div key={cit.index} className="flex items-center gap-1.5 whitespace-nowrap">
                                                <button
                                                    onClick={() => {
                                                        const matchingCitation = citationsWithMarkerPages.find((c) => c.number === cit.index);
                                                        if (matchingCitation) {
                                                            dispatch({ type: "SET_ACTIVE_CITATION", citation: matchingCitation });
                                                        }
                                                    }}
                                                    className={`inline-flex items-center justify-center min-w-[20px] h-[20px] px-1 text-[11px] font-bold rounded-full cursor-pointer transition-colors ${
                                                        isUsed
                                                            ? "text-black bg-green-400/90 hover:bg-green-400"
                                                            : "text-white bg-red-500/80 hover:bg-red-500"
                                                    }`}
                                                    title={isUsed ? `Cited in answer — view passage [${cit.index}]` : `Retrieved but not cited — view passage [${cit.index}]`}
                                                >
                                                    [{cit.index}]
                                                </button>
                                                {cit.page !== null && (
                                                    <span className="text-muted-foreground text-xs font-mono">p.{cit.page}</span>
                                                )}
                                                {idx < group.citations.length - 1 && (
                                                    <span className="text-muted-foreground/30 text-xs mx-0.5">&middot;</span>
                                                )}
                                            </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                <div className="mt-1 pl-3">
                                    <button
                                        onClick={() => toggleSource(group.sourceId, chunkIds)}
                                        className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors text-xs font-semibold"
                                    >
                                        {isExpanded ? (
                                            <ChevronDownIcon className="size-3.5" />
                                        ) : (
                                            <ChevronRightIcon className="size-3.5" />
                                        )}
                                        {isExpanded ? "Hide retrieved passages" : "Show retrieved passages"}
                                    </button>

                                    {isExpanded && (
                                        <div className="flex flex-col gap-3 mt-3 w-full max-w-full overflow-hidden">
                                            {group.citations.map((cit) => {
                                                const text = cit.text || chunkCache[cit.chunkId];
                                                return (
                                                    <div
                                                        key={cit.index}
                                                        className="flex flex-col bg-[#1e1e1e]/60 border border-white/10 rounded-md p-3 max-w-[95%]"
                                                    >
                                                        <div className="text-xs font-mono text-muted-foreground mb-1.5 flex justify-between items-center group-hover:text-foreground">
                                                            <span>[{cit.index}]{cit.page !== null ? ` p.${cit.page}` : ""}</span>
                                                        </div>
                                                        {isLoading && !text ? (
                                                            <div className="h-4 w-1/3 bg-white/10 animate-pulse rounded"></div>
                                                        ) : text ? (
                                                            <div className="break-words leading-relaxed text-[13px] text-white/80 line-clamp-none max-h-60 overflow-y-auto pr-1">
                                                                {text.length > 500 ? (
                                                                    <>
                                                                        {text.substring(0, 500)}...
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => {
                                                                                const matchingCitation = citationsWithMarkerPages.find((c) => c.number === cit.index);
                                                                                if (matchingCitation) {
                                                                                    dispatch({ type: "SET_ACTIVE_CITATION", citation: matchingCitation });
                                                                                }
                                                                            }}
                                                                            className="ml-1 text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2 cursor-pointer transition-colors"
                                                                            title="Open full passage in document viewer"
                                                                        >
                                                                            (Continues in raw passage)
                                                                        </button>
                                                                    </>
                                                                ) : (
                                                                    text
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <div className="text-red-400 text-xs italic">Failed to load snippet.</div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
