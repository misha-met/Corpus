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
  threshold: number;
  sourceIds: string[];
  onThresholdChange?: (value: number) => void;
}

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

export function PeopleDictionary({
  active = true,
  refreshNonce = 0,
  threshold,
  sourceIds,
  onThresholdChange,
}: PeopleDictionaryProps) {
  const dispatch = useAppDispatch();
  const queryClient = useQueryClient();

  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("frequency");
  const [expandedCanonical, setExpandedCanonical] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deletingMentionId, setDeletingMentionId] = useState<string | null>(null);
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false);
  const [mergeSourceCanonical, setMergeSourceCanonical] = useState<string | null>(null);
  const [mergeSearchInput, setMergeSearchInput] = useState("");
  const [mergeSearchDebounced, setMergeSearchDebounced] = useState("");
  const [mergeTargetCanonical, setMergeTargetCanonical] = useState<string | null>(null);
  const [mergeError, setMergeError] = useState<string | null>(null);
  const [isMerging, setIsMerging] = useState(false);

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
      Number(threshold.toFixed(2)),
      normalizedSourceIds,
      debouncedSearch,
      refreshNonce,
    ] as const,
    [threshold, normalizedSourceIds, debouncedSearch, refreshNonce],
  );

  const fetchPeople = useCallback(async (): Promise<PeopleListResponse> => {
    return sourceApi.getPeople(
      undefined,
      threshold,
      debouncedSearch || undefined,
      1000,
      0,
      normalizedSourceIds,
    );
  }, [normalizedSourceIds, debouncedSearch, threshold]);

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
      Number(threshold.toFixed(2)),
      normalizedSourceIds,
      refreshNonce,
    ] as const,
    [expandedCanonical, threshold, normalizedSourceIds, refreshNonce],
  );

  const fetchMentions = useCallback(async (): Promise<PersonMentionsResponse> => {
    if (!expandedCanonical) {
      return { canonical_name: "", count: 0, mentions: [] };
    }
    return sourceApi.getPeopleMentions(
      expandedCanonical,
      undefined,
      threshold,
      1000,
      0,
      normalizedSourceIds,
    );
  }, [expandedCanonical, normalizedSourceIds, threshold]);

  const mentionsQuery = useQuery<PersonMentionsResponse>({
    queryKey: mentionsQueryKey,
    queryFn: fetchMentions,
    enabled: active && normalizedSourceIds.length > 0 && !!expandedCanonical,
    placeholderData: (previousData) => previousData,
  });

  const mergeCandidatesQueryKey = useMemo(
    () => [
      "people-merge-candidates",
      Number(threshold.toFixed(2)),
      normalizedSourceIds,
      mergeSearchDebounced,
      mergeSourceCanonical,
      refreshNonce,
    ] as const,
    [threshold, normalizedSourceIds, mergeSearchDebounced, mergeSourceCanonical, refreshNonce],
  );

  const fetchMergeCandidates = useCallback(async (): Promise<PeopleListResponse> => {
    return sourceApi.getPeople(
      undefined,
      threshold,
      mergeSearchDebounced || undefined,
      200,
      0,
      normalizedSourceIds,
    );
  }, [mergeSearchDebounced, threshold, normalizedSourceIds]);

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

  const handleDeleteMention = useCallback(async (mention: PersonMention) => {
    setDeletingMentionId(mention.id);
    setActionError(null);
    try {
      await sourceApi.deletePeopleMention(mention.id);
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
      setDeletingMentionId(null);
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
      <div className="relative h-full w-full overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_120%_at_0%_0%,rgba(245,158,11,0.10)_0%,rgba(245,158,11,0.00)_48%),linear-gradient(180deg,rgba(0,0,0,0.70)_0%,rgba(0,0,0,0.78)_100%)]" />

      <div className="relative z-10 flex h-full flex-col">
        <div className="border-b border-white/10 px-4 py-3">
          <div className="flex items-center justify-between gap-3">
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
            <div className="rounded-md border border-white/15 bg-white/5 px-2 py-1 text-[10px] text-white/70">
              {(threshold * 100).toFixed(0)}% min confidence
            </div>
          </div>

          <div className="mt-2 rounded-md border border-white/15 bg-white/[0.04] px-2.5 py-2">
            <div className="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-[0.08em] text-white/65">
              <span>People confidence</span>
              <span>{Math.round(threshold * 100)}% minimum</span>
            </div>
            <input
              type="range"
              min={0.3}
              max={0.99}
              step={0.01}
              value={threshold}
              onChange={(event) => onThresholdChange?.(Number(event.target.value))}
              className="w-full accent-amber-400"
              aria-label="People confidence threshold"
            />
          </div>

          <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
            <label className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-white/45" />
              <input
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="Search people or variants"
                className="w-full rounded-md border border-white/15 bg-white/5 py-1.5 pl-7 pr-2 text-xs text-white/90 outline-none placeholder:text-white/45 focus:border-white/30"
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
              const mentions = expanded ? (mentionsQuery.data?.mentions ?? []) : [];
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
                    className="w-full px-3.5 py-3 text-left hover:bg-white/[0.03] transition-colors focus:outline-none focus:ring-1 focus:ring-white/20"
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
                      {person.roles.slice(0, 4).map((role) => (
                        <span
                          key={role}
                          className="rounded border border-amber-300/25 bg-amber-300/12 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.08em] text-amber-200"
                        >
                          {role}
                        </span>
                      ))}
                      <span className="text-[10px] text-white/60">
                        avg {(person.avg_confidence * 100).toFixed(0)}%
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
                      ) : mentions.length === 0 ? (
                        <p className="text-xs text-white/60">No mentions at this threshold/source filter.</p>
                      ) : (
                        mentions.map((mention) => (
                          <div
                            key={mention.id}
                            className="mb-2 rounded-lg border border-white/12 bg-black/25 p-2.5"
                          >
                            <div className="mb-1.5 flex items-center justify-between gap-2">
                              <p className="truncate text-[11px] font-medium text-white/85">{mention.source_id}</p>
                              <span className="rounded border border-white/18 bg-white/8 px-1.5 py-0.5 text-[10px] text-white/78">
                                {(mention.confidence * 100).toFixed(0)}%
                              </span>
                            </div>

                            <p className="text-[11px] text-white/75">{mention.context_snippet || "No context snippet"}</p>

                            <div className="mt-2 flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => handleViewMention(mention)}
                                className="inline-flex items-center gap-1 rounded-md border border-white/16 bg-white/[0.03] px-2 py-1 text-[11px] text-white/82 transition-colors hover:bg-white/10"
                              >
                                <FileText className="h-3.5 w-3.5" />
                                View in document
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDeleteMention(mention)}
                                disabled={deletingMentionId === mention.id}
                                className="inline-flex items-center gap-1 rounded-md border border-red-400/35 bg-red-500/[0.08] px-2 py-1 text-[11px] text-red-300 transition-colors hover:bg-red-500/[0.16] disabled:opacity-50"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                                Delete
                              </button>
                            </div>
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
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-white/45" />
              <input
                value={mergeSearchInput}
                onChange={(event) => setMergeSearchInput(event.target.value)}
                placeholder="Search canonical names"
                autoFocus
                className="w-full rounded-md border border-white/15 bg-white/5 py-1.5 pl-7 pr-2 text-xs text-white/90 outline-none placeholder:text-white/45 focus:border-white/30"
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
