"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRightLeft, FileText, Loader2, Search, Trash2, UserRound } from "lucide-react";

import {
  sourceApi,
  type PeopleListResponse,
  type PersonMention,
  type PersonMentionsResponse,
  type PersonSummary,
} from "@/lib/api-client";
import { useAppDispatch } from "@/context/app-context";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type SortMode = "frequency" | "alphabetical";

interface PeopleDictionaryProps {
  active?: boolean;
  refreshNonce?: number;
  sourceIds: string[];
}

interface MentionDisplayGroup {
  key: string;
  sourceId: string;
  rawName: string;
  pageLabel: string;
  mentions: PersonMention[];
  representative: PersonMention;
}

const UNKNOWN_PAGE_LABEL = "Unknown page";

function normalizeSourceIds(sourceIds: string[]): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const raw of sourceIds) {
    const sid = String(raw).trim();
    if (!sid || seen.has(sid)) continue;
    seen.add(sid);
    normalized.push(sid);
  }
  return normalized;
}

function chunkLookupKey(sourceId: string, chunkId: string): string {
  return `${sourceId}::${chunkId}`;
}

function formatChunkPageLabel(displayPage: string | null | undefined, pageNumber: number | null | undefined): string {
  if (displayPage && String(displayPage).trim().length > 0) {
    const normalized = String(displayPage).trim();
    if (/^(page|pages|p\.|pp\.)\s+/i.test(normalized)) {
      return normalized;
    }
    if (/^\d+$/.test(normalized)) {
      return `Page ${normalized}`;
    }
    if (/^\d+\s*-\s*\d+$/.test(normalized)) {
      return `Pages ${normalized}`;
    }
    return `Page ${normalized}`;
  }
  if (typeof pageNumber === "number" && Number.isFinite(pageNumber) && pageNumber > 0) {
    return `Page ${pageNumber}`;
  }
  return UNKNOWN_PAGE_LABEL;
}

export function PeopleDictionary({
  active = true,
  refreshNonce = 0,
  sourceIds,
}: PeopleDictionaryProps) {
  const dispatch = useAppDispatch();
  const queryClient = useQueryClient();

  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("frequency");
  const [expandedCanonical, setExpandedCanonical] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deletingMentionKey, setDeletingMentionKey] = useState<string | null>(null);
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false);
  const [mergeSourceCanonical, setMergeSourceCanonical] = useState<string | null>(null);
  const [mergeSearchInput, setMergeSearchInput] = useState("");
  const [mergeSearchDebounced, setMergeSearchDebounced] = useState("");
  const [mergeTargetCanonical, setMergeTargetCanonical] = useState<string | null>(null);
  const [mergeError, setMergeError] = useState<string | null>(null);
  const [isMerging, setIsMerging] = useState(false);
  const [chunkPageLabels, setChunkPageLabels] = useState<Record<string, string>>({});
  const [isResolvingPages, setIsResolvingPages] = useState(false);

  const normalizedSourceIds = useMemo(
    () => normalizeSourceIds(sourceIds),
    [sourceIds],
  );

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
    }, 220);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setMergeSearchDebounced(mergeSearchInput.trim());
    }, 180);
    return () => clearTimeout(timer);
  }, [mergeSearchInput]);

  const listQueryKey = useMemo(
    () => [
      "people-list",
      normalizedSourceIds,
      debouncedSearch,
      refreshNonce,
    ] as const,
    [normalizedSourceIds, debouncedSearch, refreshNonce],
  );

  const fetchPeople = useCallback(async (): Promise<PeopleListResponse> => {
    return sourceApi.getPeople(
      undefined,
      0.0,
      debouncedSearch || undefined,
      1000,
      0,
      normalizedSourceIds,
    );
  }, [normalizedSourceIds, debouncedSearch]);

  const peopleQuery = useQuery<PeopleListResponse>({
    queryKey: listQueryKey,
    queryFn: fetchPeople,
    enabled: active && normalizedSourceIds.length > 0,
    placeholderData: (previousData) => previousData,
  });

  const mentionsQueryKey = useMemo(
    () => [
      "people-mentions",
      expandedCanonical,
      normalizedSourceIds,
      refreshNonce,
    ] as const,
    [expandedCanonical, normalizedSourceIds, refreshNonce],
  );

  const fetchMentions = useCallback(async (): Promise<PersonMentionsResponse> => {
    if (!expandedCanonical) {
      return { canonical_name: "", count: 0, mentions: [] };
    }
    return sourceApi.getPeopleMentions(
      expandedCanonical,
      undefined,
      0.0,
      1000,
      0,
      normalizedSourceIds,
    );
  }, [expandedCanonical, normalizedSourceIds]);

  const mentionsQuery = useQuery<PersonMentionsResponse>({
    queryKey: mentionsQueryKey,
    queryFn: fetchMentions,
    enabled: active && normalizedSourceIds.length > 0 && !!expandedCanonical,
    placeholderData: (previousData) => previousData,
  });

  const mergeCandidatesQueryKey = useMemo(
    () => [
      "people-merge-candidates",
      normalizedSourceIds,
      mergeSearchDebounced,
      mergeSourceCanonical,
      refreshNonce,
    ] as const,
    [normalizedSourceIds, mergeSearchDebounced, mergeSourceCanonical, refreshNonce],
  );

  const fetchMergeCandidates = useCallback(async (): Promise<PeopleListResponse> => {
    return sourceApi.getPeople(
      undefined,
      0.0,
      mergeSearchDebounced || undefined,
      200,
      0,
      normalizedSourceIds,
    );
  }, [mergeSearchDebounced, normalizedSourceIds]);

  const mergeCandidatesQuery = useQuery<PeopleListResponse>({
    queryKey: mergeCandidatesQueryKey,
    queryFn: fetchMergeCandidates,
    enabled: active && mergeDialogOpen && normalizedSourceIds.length > 0,
    placeholderData: (previousData) => previousData,
  });

  const mergeCandidates = useMemo(() => {
    const source = (mergeSourceCanonical ?? "").trim().toLowerCase();
    return (mergeCandidatesQuery.data?.people ?? []).filter(
      (item) => item.canonical_name.trim().toLowerCase() !== source,
    );
  }, [mergeCandidatesQuery.data?.people, mergeSourceCanonical]);

  const openMergeDialog = useCallback((sourceCanonicalName: string) => {
    setMergeSourceCanonical(sourceCanonicalName);
    setMergeSearchInput("");
    setMergeSearchDebounced("");
    setMergeTargetCanonical(null);
    setMergeError(null);
    setMergeDialogOpen(true);
  }, []);

  const closeMergeDialog = useCallback(() => {
    setMergeDialogOpen(false);
    setMergeSearchInput("");
    setMergeSearchDebounced("");
    setMergeTargetCanonical(null);
    setMergeError(null);
  }, []);

  useEffect(() => {
    if (!mergeTargetCanonical) return;
    if (mergeCandidates.some((item) => item.canonical_name === mergeTargetCanonical)) return;
    setMergeTargetCanonical(null);
  }, [mergeCandidates, mergeTargetCanonical]);

  const handleMergeCanonical = useCallback(async () => {
    const sourceCanonical = mergeSourceCanonical?.trim() ?? "";
    const targetCanonical = mergeTargetCanonical?.trim() ?? "";
    if (!sourceCanonical || !targetCanonical) {
      setMergeError("Select a target canonical name to merge into.");
      return;
    }

    setIsMerging(true);
    setMergeError(null);
    try {
      const result = await sourceApi.mergePeopleCanonical(sourceCanonical, targetCanonical);
      if (result.merged_count === 0) {
        setMergeError("No mentions were merged. The source name may already be empty.");
        return;
      }

      await queryClient.invalidateQueries({ queryKey: ["people-list"] });
      await queryClient.invalidateQueries({ queryKey: ["people-mentions"] });
      await queryClient.fetchQuery({
        queryKey: listQueryKey,
        queryFn: fetchPeople,
        staleTime: 0,
      });

      if (expandedCanonical) {
        await queryClient.fetchQuery({
          queryKey: mentionsQueryKey,
          queryFn: fetchMentions,
          staleTime: 0,
        });
      }

      setExpandedCanonical((prev) => (prev === sourceCanonical ? targetCanonical : prev));
      closeMergeDialog();
    } catch (err) {
      setMergeError(err instanceof Error ? err.message : "Failed to merge names");
    } finally {
      setIsMerging(false);
    }
  }, [
    closeMergeDialog,
    expandedCanonical,
    fetchMentions,
    fetchPeople,
    listQueryKey,
    mentionsQueryKey,
    mergeSourceCanonical,
    mergeTargetCanonical,
    queryClient,
  ]);

  useEffect(() => {
    if (!expandedCanonical) return;
    const list = peopleQuery.data?.people ?? [];
    if (!list.some((item) => item.canonical_name === expandedCanonical)) {
      setExpandedCanonical(null);
    }
  }, [expandedCanonical, peopleQuery.data?.people]);

  useEffect(() => {
    setActionError(null);
  }, [listQueryKey]);

  useEffect(() => {
    setChunkPageLabels({});
  }, [refreshNonce, normalizedSourceIds]);

  useEffect(() => {
    if (!expandedCanonical) return;
    const mentions = mentionsQuery.data?.mentions ?? [];
    if (mentions.length === 0) return;

    const missingBySource = new Map<string, Set<string>>();
    for (const mention of mentions) {
      const lookup = chunkLookupKey(mention.source_id, mention.chunk_id);
      if (chunkPageLabels[lookup]) continue;
      const chunkSet = missingBySource.get(mention.source_id) ?? new Set<string>();
      chunkSet.add(mention.chunk_id);
      missingBySource.set(mention.source_id, chunkSet);
    }

    if (missingBySource.size === 0) return;

    let cancelled = false;
    setIsResolvingPages(true);

    const resolvePages = async (): Promise<void> => {
      const updates: Record<string, string> = {};
      try {
        for (const [sourceId, chunkSet] of missingBySource.entries()) {
          const chunkIds = Array.from(chunkSet);
          if (chunkIds.length === 0) continue;

          const response = await sourceApi.getChunks(sourceId, chunkIds);
          for (const chunk of response.chunks) {
            updates[chunkLookupKey(sourceId, chunk.chunk_id)] = formatChunkPageLabel(
              chunk.display_page,
              chunk.page_number,
            );
          }

          for (const chunkId of chunkIds) {
            const lookup = chunkLookupKey(sourceId, chunkId);
            if (!updates[lookup]) {
              updates[lookup] = UNKNOWN_PAGE_LABEL;
            }
          }
        }
      } catch {
        for (const [sourceId, chunkSet] of missingBySource.entries()) {
          for (const chunkId of chunkSet) {
            updates[chunkLookupKey(sourceId, chunkId)] = UNKNOWN_PAGE_LABEL;
          }
        }
      } finally {
        if (!cancelled) {
          if (Object.keys(updates).length > 0) {
            setChunkPageLabels((prev) => ({ ...prev, ...updates }));
          }
          setIsResolvingPages(false);
        }
      }
    };

    void resolvePages();

    return () => {
      cancelled = true;
    };
  }, [expandedCanonical, mentionsQuery.data?.mentions, chunkPageLabels]);

  const people = useMemo(() => {
    const rows = [...(peopleQuery.data?.people ?? [])];
    if (sortMode === "alphabetical") {
      rows.sort((a, b) => a.canonical_name.localeCompare(b.canonical_name));
      return rows;
    }
    rows.sort((a, b) => {
      if (b.mention_count !== a.mention_count) return b.mention_count - a.mention_count;
      return a.canonical_name.localeCompare(b.canonical_name);
    });
    return rows;
  }, [peopleQuery.data?.people, sortMode]);

  const error = actionError ?? (peopleQuery.error instanceof Error ? peopleQuery.error.message : null);

  const mentionGroups = useMemo((): MentionDisplayGroup[] => {
    const mentions = mentionsQuery.data?.mentions ?? [];
    const grouped = new Map<string, MentionDisplayGroup>();

    for (const mention of mentions) {
      const lookup = chunkLookupKey(mention.source_id, mention.chunk_id);
      const pageLabel = chunkPageLabels[lookup] ?? UNKNOWN_PAGE_LABEL;
      const pageBucket =
        pageLabel === UNKNOWN_PAGE_LABEL
          ? `chunk:${mention.chunk_id}`
          : pageLabel.trim().toLowerCase();
      const key = `${mention.source_id}::${mention.raw_name.trim().toLowerCase()}::${pageBucket}`;

      const existing = grouped.get(key);
      if (existing) {
        existing.mentions.push(mention);
        continue;
      }

      grouped.set(key, {
        key,
        sourceId: mention.source_id,
        rawName: mention.raw_name,
        pageLabel,
        mentions: [mention],
        representative: mention,
      });
    }

    for (const group of grouped.values()) {
      group.mentions.sort((a, b) => {
        const chunkCmp = a.chunk_id.localeCompare(b.chunk_id);
        if (chunkCmp !== 0) return chunkCmp;
        return a.id.localeCompare(b.id);
      });
      group.representative = group.mentions[0];
    }

    return Array.from(grouped.values()).sort((a, b) => {
      const sourceCmp = a.sourceId.localeCompare(b.sourceId);
      if (sourceCmp !== 0) return sourceCmp;
      const pageCmp = a.pageLabel.localeCompare(b.pageLabel);
      if (pageCmp !== 0) return pageCmp;
      return a.rawName.localeCompare(b.rawName);
    });
  }, [mentionsQuery.data?.mentions, chunkPageLabels]);

  const getMentionPageLabel = useCallback((mention: PersonMention): string => {
    const lookup = chunkLookupKey(mention.source_id, mention.chunk_id);
    return chunkPageLabels[lookup] ?? UNKNOWN_PAGE_LABEL;
  }, [chunkPageLabels]);

  const handleViewMention = useCallback(async (mention: PersonMention) => {
    const chunk = await sourceApi.getChunk(mention.source_id, mention.chunk_id);
    dispatch({
      type: "SET_ACTIVE_CITATION",
      citation: {
        number: 0,
        source_id: mention.source_id,
        chunk_id: mention.chunk_id,
        page: chunk.page_number ?? null,
        header_path: chunk.header_path,
        chunk_text: chunk.chunk_text,
        highlight_text: mention.raw_name,
        highlight_scope_text: chunk.chunk_text,
      },
    });
  }, [dispatch]);

  const handleDeleteMentionGroup = useCallback(async (groupKey: string, mentions: PersonMention[]) => {
    const mentionIds = Array.from(new Set(mentions.map((mention) => mention.id)));
    if (mentionIds.length === 0) return;

    setDeletingMentionKey(groupKey);
    setActionError(null);
    try {
      await Promise.all(mentionIds.map((mentionId) => sourceApi.deletePeopleMention(mentionId)));
      await queryClient.invalidateQueries({ queryKey: listQueryKey, exact: true });
      await queryClient.fetchQuery({
        queryKey: listQueryKey,
        queryFn: fetchPeople,
        staleTime: 0,
      });

      if (expandedCanonical) {
        await queryClient.invalidateQueries({ queryKey: mentionsQueryKey, exact: true });
        await queryClient.fetchQuery({
          queryKey: mentionsQueryKey,
          queryFn: fetchMentions,
          staleTime: 0,
        });
      }
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to delete people mention");
    } finally {
      setDeletingMentionKey(null);
    }
  }, [expandedCanonical, fetchMentions, fetchPeople, listQueryKey, mentionsQueryKey, queryClient]);

  if (normalizedSourceIds.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-gray-300">
        <UserRound className="h-9 w-9 text-gray-500" />
        <p>Select at least one source to browse people mentions.</p>
      </div>
    );
  }

  if (peopleQuery.isLoading && !peopleQuery.data) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-400">
        Loading people dictionary...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-red-300">
        {error}
      </div>
    );
  }

  return (
    <>
      <div className="relative h-full w-full overflow-hidden bg-black/62 backdrop-blur-xl">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.05)_0%,rgba(255,255,255,0.00)_22%)]" />

      <div className="relative z-10 flex h-full flex-col">
        <div className="border-b border-white/10 px-4 py-3">
          <div>
            <div>
              <p className="text-sm font-semibold tracking-tight text-white/92">People Dictionary</p>
              <div className="flex items-center gap-2 text-xs text-white/58">
                <p>
                  {peopleQuery.data?.count ?? 0} canonical entr{(peopleQuery.data?.count ?? 0) === 1 ? "y" : "ies"}
                </p>
                {peopleQuery.isFetching && (
                  <span className="inline-flex items-center gap-1 text-white/45">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Updating
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-2 grid grid-cols-[1fr_auto] gap-2">
            <label className="flex h-9 items-center rounded-md border border-white/15 bg-white/5 px-2.5 focus-within:border-white/30">
              <Search className="h-4 w-4 shrink-0 text-white/45" />
              <input
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="Search people or variants"
                className="h-full w-full bg-transparent px-2 text-xs text-white/90 outline-none placeholder:text-white/45"
              />
            </label>

            <div className="flex items-center gap-1 rounded-md border border-white/15 bg-white/[0.04] p-1">
              <button
                type="button"
                onClick={() => setSortMode("frequency")}
                className={`rounded px-2 py-1 text-[11px] transition-colors ${
                  sortMode === "frequency"
                    ? "bg-white/18 text-white"
                    : "text-white/65 hover:bg-white/10 hover:text-white"
                }`}
              >
                Frequency
              </button>
              <button
                type="button"
                onClick={() => setSortMode("alphabetical")}
                className={`rounded px-2 py-1 text-[11px] transition-colors ${
                  sortMode === "alphabetical"
                    ? "bg-white/18 text-white"
                    : "text-white/65 hover:bg-white/10 hover:text-white"
                }`}
              >
                A-Z
              </button>
            </div>
          </div>
        </div>

        {people.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-gray-300">
            <UserRound className="h-9 w-9 text-gray-500" />
            <p>No people indexed yet. Enable &quot;Index people names&quot; when ingesting documents.</p>
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            {people.map((person: PersonSummary) => {
              const expanded = expandedCanonical === person.canonical_name;
              return (
                <div
                  key={person.canonical_name}
                  className="mb-3 rounded-xl border border-white/12 bg-white/[0.035] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
                >
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => setExpandedCanonical((prev) => (prev === person.canonical_name ? null : person.canonical_name))}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      setExpandedCanonical((prev) => (prev === person.canonical_name ? null : person.canonical_name));
                    }}
                    className="w-full cursor-pointer px-3.5 py-3 text-left focus:outline-none"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-sm font-semibold text-white/92">{person.canonical_name}</p>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            openMergeDialog(person.canonical_name);
                          }}
                          className="inline-flex items-center gap-1 rounded-md border border-amber-300/30 bg-amber-400/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-200 transition-colors hover:bg-amber-400/18"
                          title={`Merge ${person.canonical_name} into another name`}
                        >
                          <ArrowRightLeft className="h-3 w-3" />
                          Merge
                        </button>
                        <span className="rounded-md border border-white/18 bg-white/8 px-2 py-0.5 text-[10px] font-medium text-white/82">
                          {person.mention_count} mentions
                        </span>
                      </div>
                    </div>

                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span className="rounded border border-amber-300/25 bg-amber-300/12 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.08em] text-amber-200">
                        mentioned
                      </span>
                    </div>

                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      {person.source_ids.slice(0, 3).map((sid) => (
                        <span key={sid} className="rounded border border-white/14 bg-white/6 px-1.5 py-0.5 text-[10px] text-white/70">
                          {sid}
                        </span>
                      ))}
                      {person.source_ids.length > 3 && (
                        <span className="text-[10px] text-white/60">+{person.source_ids.length - 3} more</span>
                      )}
                    </div>
                  </div>

                  {expanded && (
                    <div className="border-t border-white/10 px-3.5 py-3">
                      {mentionsQuery.isLoading || mentionsQuery.isFetching ? (
                        <p className="text-xs text-white/60">Loading mentions...</p>
                      ) : mentionGroups.length === 0 ? (
                        <p className="text-xs text-white/60">No mentions for this person in the selected sources.</p>
                      ) : (
                        mentionGroups.map((group) => (
                          <div
                            key={group.key}
                            className="mb-2 rounded-lg border border-white/12 bg-black/25 p-2.5"
                          >
                            <div className="mb-1.5 flex items-center justify-between gap-2">
                              <p className="truncate text-[11px] font-medium text-white/85">{group.sourceId}</p>
                              <div className="flex items-center gap-1">
                                <span className="rounded border border-white/18 bg-white/8 px-1.5 py-0.5 text-[10px] text-white/78">
                                  Location: {group.pageLabel}
                                </span>
                                <span className="rounded border border-amber-300/25 bg-amber-300/12 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.08em] text-amber-200">
                                  mentioned
                                </span>
                              </div>
                            </div>

                            <p className="text-[11px] text-white/75">{group.rawName}</p>
                            {group.mentions.length > 1 && (
                              <p className="mt-1 text-[10px] text-white/55">
                                {group.mentions.length} occurrences on this source/page
                              </p>
                            )}

                            {isResolvingPages && group.pageLabel === UNKNOWN_PAGE_LABEL && (
                              <p className="mt-1 text-[10px] text-white/55">Resolving page number...</p>
                            )}

                            {group.mentions.length > 1 ? (
                              <div className="mt-2 space-y-1.5">
                                {group.mentions.map((mention, index) => {
                                  const mentionDeleteKey = `mention:${mention.id}`;
                                  const mentionPageLabel = getMentionPageLabel(mention);
                                  return (
                                    <div
                                      key={mention.id}
                                      className="rounded-md border border-white/10 bg-black/35 px-2 py-1.5"
                                    >
                                      <div className="flex items-center justify-between gap-2">
                                        <div className="min-w-0">
                                          <span className="block truncate text-[10px] text-white/68">
                                            Occurrence {index + 1}
                                          </span>
                                          <span className="block truncate text-[10px] text-white/58">
                                            Location: {mentionPageLabel}
                                          </span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                          <button
                                            type="button"
                                            onClick={() => handleViewMention(mention)}
                                            className="inline-flex items-center gap-1 rounded-md border border-white/16 bg-white/[0.03] px-1.5 py-1 text-[10px] text-white/82 transition-colors hover:bg-white/10"
                                          >
                                            <FileText className="h-3 w-3" />
                                            View
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => handleDeleteMentionGroup(mentionDeleteKey, [mention])}
                                            disabled={deletingMentionKey === mentionDeleteKey}
                                            className="inline-flex items-center gap-1 rounded-md border border-red-400/35 bg-red-500/[0.08] px-1.5 py-1 text-[10px] text-red-300 transition-colors hover:bg-red-500/[0.16] disabled:opacity-50"
                                          >
                                            <Trash2 className="h-3 w-3" />
                                            {deletingMentionKey === mentionDeleteKey ? "Deleting..." : "Delete"}
                                          </button>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}

                                <button
                                  type="button"
                                  onClick={() => handleDeleteMentionGroup(group.key, group.mentions)}
                                  disabled={deletingMentionKey === group.key}
                                  className="inline-flex items-center gap-1 rounded-md border border-red-400/35 bg-red-500/[0.08] px-2 py-1 text-[11px] text-red-300 transition-colors hover:bg-red-500/[0.16] disabled:opacity-50"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                  {deletingMentionKey === group.key
                                    ? "Deleting all..."
                                    : `Delete all ${group.mentions.length}`}
                                </button>
                              </div>
                            ) : (
                              <div className="mt-2 flex items-center gap-2">
                                <span className="truncate text-[10px] text-white/58">
                                  Occurrence 1 - Location: {getMentionPageLabel(group.representative)}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => handleViewMention(group.representative)}
                                  className="inline-flex items-center gap-1 rounded-md border border-white/16 bg-white/[0.03] px-2 py-1 text-[11px] text-white/82 transition-colors hover:bg-white/10"
                                >
                                  <FileText className="h-3.5 w-3.5" />
                                  View in document
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteMentionGroup(group.key, group.mentions)}
                                  disabled={deletingMentionKey === group.key}
                                  className="inline-flex items-center gap-1 rounded-md border border-red-400/35 bg-red-500/[0.08] px-2 py-1 text-[11px] text-red-300 transition-colors hover:bg-red-500/[0.16] disabled:opacity-50"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                  {deletingMentionKey === group.key ? "Deleting..." : "Delete"}
                                </button>
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      </div>

      <Dialog
        open={mergeDialogOpen}
        onOpenChange={(open) => {
          if (open) {
            setMergeDialogOpen(true);
            return;
          }
          closeMergeDialog();
        }}
      >
        <DialogContent className="max-w-md border border-white/18 bg-[#111317] text-white sm:rounded-xl [&>button]:border [&>button]:border-white/20 [&>button]:bg-black/45 [&>button]:text-white/80 [&>button]:opacity-100 [&>button]:hover:text-white">
          <DialogHeader>
            <DialogTitle className="text-base text-white">Merge Canonical Name</DialogTitle>
            <DialogDescription className="text-xs text-white/65">
              Move all mentions from one canonical entry into another.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3">
            <div className="rounded-md border border-white/14 bg-white/[0.03] px-2.5 py-2 text-xs text-white/80">
              <span className="text-white/55">Source:</span>{" "}
              <span className="font-semibold text-white/95">{mergeSourceCanonical ?? "No source selected"}</span>
            </div>

            <label className="block text-[11px] uppercase tracking-[0.08em] text-white/55">
              Target canonical name
            </label>
            <label className="relative block">
              <span className="pointer-events-none absolute inset-y-0 left-2.5 flex items-center text-white/45">
                <Search className="h-3.5 w-3.5" />
              </span>
              <input
                value={mergeSearchInput}
                onChange={(event) => setMergeSearchInput(event.target.value)}
                placeholder="Search canonical names"
                autoFocus
                className="w-full rounded-md border border-white/15 bg-white/5 py-1.5 pl-8 pr-2 text-xs text-white/90 outline-none placeholder:text-white/45 focus:border-white/30"
              />
            </label>

            <div className="max-h-52 overflow-y-auto rounded-md border border-white/14 bg-black/30">
              {mergeCandidatesQuery.isLoading && !mergeCandidatesQuery.data ? (
                <div className="px-3 py-2 text-xs text-white/60">Loading names...</div>
              ) : mergeCandidates.length === 0 ? (
                <div className="px-3 py-2 text-xs text-white/60">No matching canonical names found.</div>
              ) : (
                mergeCandidates.map((candidate) => {
                  const isSelected = candidate.canonical_name === mergeTargetCanonical;
                  return (
                    <button
                      key={candidate.canonical_name}
                      type="button"
                      onClick={() => setMergeTargetCanonical(candidate.canonical_name)}
                      className={`flex w-full items-center justify-between px-3 py-2 text-left text-xs transition-colors ${
                        isSelected
                          ? "bg-amber-400/18 text-amber-100"
                          : "text-white/82 hover:bg-white/8"
                      }`}
                    >
                      <span className="truncate">{candidate.canonical_name}</span>
                      <span className="ml-2 shrink-0 text-[10px] text-white/55">
                        {candidate.mention_count} mentions
                      </span>
                    </button>
                  );
                })
              )}
            </div>

            {mergeError && (
              <p className="text-xs text-red-300">{mergeError}</p>
            )}
          </div>

          <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
            <button
              type="button"
              onClick={closeMergeDialog}
              disabled={isMerging}
              className="rounded-md border border-white/18 bg-white/[0.04] px-3 py-1.5 text-xs text-white/82 transition-colors hover:bg-white/10 disabled:opacity-60"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void handleMergeCanonical()}
              disabled={!mergeTargetCanonical || isMerging}
              className="inline-flex items-center gap-1.5 rounded-md border border-amber-300/35 bg-amber-400/15 px-3 py-1.5 text-xs font-medium text-amber-100 transition-colors hover:bg-amber-400/22 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isMerging && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {isMerging ? "Merging..." : "Merge Into Selected"}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
