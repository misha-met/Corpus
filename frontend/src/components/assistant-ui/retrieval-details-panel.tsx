"use client";

/**
 * RetrievalDetailsPanel — collapsible panel showing per-query retrieval
 * diagnostics: timing, score distribution, candidate funnel, and budget.
 *
 * Layout:
 *  [collapsed] single summary line
 *  [expanded]  4 sections:
 *    1. Timing breakdown — horizontal stacked bar (inline SVG)
 *    2. Score distribution — mini histogram (inline SVG) + threshold line
 *    3. Candidate funnel — scrollable compact table, 20-row limit with toggle
 *    4. Context budget — horizontal meter bar with utilisation numbers
 */

import { useAppState } from "@/context/app-context";
import type { CandidateDecision, RetrievalDetails } from "@/lib/event-parser";
import { type FC, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtMs(ms: number | undefined): string {
  if (!ms) return "—";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function fmtPct(pct: number | undefined): string {
  if (pct == null) return "—";
  return `${Math.round(pct)}%`;
}

const STATUS_COLORS: Record<string, string> = {
  kept: "bg-blue-500/80",
  filtered: "bg-zinc-500/60",
  deduplicated: "bg-yellow-500/70",
  budget_cut: "bg-orange-500/70",
};

const STATUS_TEXT: Record<string, string> = {
  kept: "kept",
  filtered: "filtered",
  deduplicated: "deduped",
  budget_cut: "budget cut",
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TimingBar({ timings }: { timings: RetrievalDetails["timings"] }) {
  const stages = [
    { key: "query_embedding_ms", label: "Embed", color: "#60a5fa" },
    { key: "hybrid_search_ms", label: "Search", color: "#818cf8" },
    { key: "rerank_ms", label: "Rerank", color: "#a78bfa" },
    { key: "dedup_ms", label: "Dedup", color: "#34d399" },
    { key: "budget_packing_ms", label: "Budget", color: "#f59e0b" },
  ] as const;

  const values = stages.map((s) => ({ ...s, ms: timings[s.key] ?? 0 }));
  const total = values.reduce((acc, s) => acc + s.ms, 0) || 1;

  return (
    <div>
      <div className="mb-1 text-xs text-muted-foreground">
        Stage timings — total {fmtMs(timings["total_ms"])}
      </div>
      <div className="flex h-4 w-full overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.06)" }}>
        {values.map((s) => (
          <div
            key={s.key}
            title={`${s.label}: ${fmtMs(s.ms)}`}
            style={{ width: `${(s.ms / total) * 100}%`, background: s.color, minWidth: s.ms > 0 ? 2 : 0 }}
          />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        {values.filter((s) => s.ms > 0).map((s) => (
          <span key={s.key} className="flex items-center gap-1">
            <span className="inline-block size-2 rounded-full shrink-0" style={{ background: s.color }} />
            {s.label} {fmtMs(s.ms)}
          </span>
        ))}
      </div>
    </div>
  );
}

function ScoreHistogram({ dist }: { dist: RetrievalDetails["score_distribution"] }) {
  const BIN_COUNT = 12;
  const min = dist.min ?? 0;
  const max = dist.max ?? 1;
  const range = max - min || 1;
  const threshold = dist.threshold ?? 0;
  // Build bins from percentile bands — approximate
  const samplePoints = [
    dist.min, dist.percentile_10, dist.percentile_25, dist.percentile_50,
    dist.percentile_75, dist.percentile_90, dist.max,
  ].filter((v) => v != null) as number[];

  // Simple histogram: count how many sample points fall into each bin
  const bins = Array(BIN_COUNT).fill(0) as number[];
  for (const v of samplePoints) {
    const idx = Math.min(BIN_COUNT - 1, Math.floor(((v - min) / range) * BIN_COUNT));
    bins[idx]++;
  }
  const maxBin = Math.max(...bins, 1);
  const SVG_W = 220;
  const SVG_H = 48;
  const barW = SVG_W / BIN_COUNT;

  const thresholdX = ((threshold - min) / range) * SVG_W;

  return (
    <div>
      <div className="mb-1 text-xs text-muted-foreground">
        Score distribution — n={dist.n}, mean={dist.mean?.toFixed(3)}, σ={dist.std?.toFixed(3)}
      </div>
      <svg width={SVG_W} height={SVG_H} className="overflow-visible">
        {bins.map((count, i) => {
          const barH = (count / maxBin) * (SVG_H - 4);
          return (
            <rect
              key={i}
              x={i * barW + 1}
              y={SVG_H - 4 - barH}
              width={barW - 2}
              height={barH}
              fill="rgba(99,102,241,0.6)"
            />
          );
        })}
        {/* Threshold line */}
        {thresholdX >= 0 && thresholdX <= SVG_W && (
          <line
            x1={thresholdX} y1={0} x2={thresholdX} y2={SVG_H}
            stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="3,2"
          />
        )}
        {/* Axis labels */}
        <text x={0} y={SVG_H + 10} fontSize={9} fill="rgba(255,255,255,0.4)">{min.toFixed(2)}</text>
        <text x={SVG_W} y={SVG_H + 10} fontSize={9} fill="rgba(255,255,255,0.4)" textAnchor="end">{max.toFixed(2)}</text>
      </svg>
      <div className="mt-1 flex items-center gap-1 text-[10px] text-muted-foreground">
        <span className="inline-block h-[2px] w-4 rounded" style={{ background: "#f59e0b" }} />
        threshold {threshold.toFixed(3)}
      </div>
    </div>
  );
}

const TABLE_ROW_LIMIT = 20;

function CandidateTable({ candidates }: { candidates: CandidateDecision[] }) {
  const [showAll, setShowAll] = useState(false);
  const displayed = showAll ? candidates : candidates.slice(0, TABLE_ROW_LIMIT);

  return (
    <div>
      <div className="mb-1 text-xs text-muted-foreground">
        Candidates ({candidates.length} total — {candidates.filter((c) => c.status === "kept").length} kept)
      </div>
      <div className="overflow-x-auto rounded border border-white/10">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-white/10 text-left text-muted-foreground">
              <th className="px-2 py-1 font-normal">#</th>
              <th className="px-2 py-1 font-normal">Source</th>
              <th className="px-2 py-1 font-normal">Pg</th>
              <th className="px-2 py-1 font-normal">Score</th>
              <th className="px-2 py-1 font-normal">%ile</th>
              <th className="px-2 py-1 font-normal">Status</th>
              <th className="px-2 py-1 font-normal">Preview</th>
            </tr>
          </thead>
          <tbody>
            {displayed.map((c) => (
              <tr key={c.chunk_id} className="border-b border-white/5 hover:bg-white/5">
                <td className="px-2 py-0.5 text-muted-foreground">{c.rank}</td>
                <td className="px-2 py-0.5 max-w-[90px] truncate text-foreground/80" title={c.source_id}>
                  {c.source_id.replace(/_/g, " ")}
                </td>
                <td className="px-2 py-0.5 text-muted-foreground">{c.page ?? "—"}</td>
                <td className="px-2 py-0.5 tabular-nums text-foreground/80">{c.score.toFixed(3)}</td>
                <td className="px-2 py-0.5 tabular-nums text-muted-foreground">{c.percentile.toFixed(0)}</td>
                <td className="px-2 py-0.5">
                  <span
                    className={`inline-block rounded px-1 py-px text-[10px] font-medium text-white/90 ${STATUS_COLORS[c.status] ?? "bg-zinc-600/60"}`}
                    title={c.reason || undefined}
                  >
                    {STATUS_TEXT[c.status] ?? c.status}
                  </span>
                </td>
                <td className="px-2 py-0.5 max-w-[140px] truncate text-muted-foreground" title={c.text_preview}>
                  {c.text_preview}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!showAll && candidates.length > TABLE_ROW_LIMIT && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-1 text-[10px] text-blue-400 hover:underline"
        >
          Show all {candidates.length} candidates
        </button>
      )}
    </div>
  );
}

function BudgetMeter({ budget }: { budget: RetrievalDetails["budget"] }) {
  const pct = budget?.utilization_pct ?? 0;
  return (
    <div>
      <div className="mb-1 text-xs text-muted-foreground">
        Context budget — {budget?.used_tokens?.toLocaleString() ?? "?"} / {budget?.budget_tokens?.toLocaleString() ?? "?"} tokens ({fmtPct(pct)})
      </div>
      <div className="h-3 w-full overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${Math.min(100, pct)}%`,
            background: pct > 90 ? "#f59e0b" : "#60a5fa",
          }}
        />
      </div>
      <div className="mt-1 flex gap-4 text-[10px] text-muted-foreground">
        <span>{budget?.docs_packed ?? 0} packed</span>
        {(budget?.docs_skipped ?? 0) > 0 && <span className="text-orange-400">{budget.docs_skipped} skipped</span>}
        {(budget?.docs_truncated ?? 0) > 0 && <span className="text-yellow-400">{budget.docs_truncated} truncated</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export const RetrievalDetailsPanel: FC = () => {
  const { retrievalDetails } = useAppState();
  const [open, setOpen] = useState(false);

  if (!retrievalDetails) return null;

  const d = retrievalDetails;
  const nKept = d.candidates.filter((c) => c.status === "kept").length;
  const nTotal = d.candidates.length;
  const totalMs = d.timings["total_ms"];
  const threshold = d.threshold_info?.threshold_value;
  const budgetPct = d.budget?.utilization_pct;

  const summaryLine = [
    `${nKept} / ${nTotal} chunks kept`,
    d.source_diversity?.distinct_sources != null && `${d.source_diversity.distinct_sources} sources`,
    threshold != null && `Threshold: ${threshold.toFixed(2)}`,
    budgetPct != null && `${fmtPct(budgetPct)} budget`,
    totalMs != null && fmtMs(totalMs),
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="mt-1">
      <Collapsible.Trigger className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-white/5 hover:text-foreground/80 transition-colors">
        {open ? <ChevronDown className="size-3 shrink-0" /> : <ChevronRight className="size-3 shrink-0" />}
        <span className="font-medium text-foreground/60">Retrieval details</span>
        <span className="ml-1 truncate">{summaryLine}</span>
      </Collapsible.Trigger>

      <Collapsible.Content className="mt-1 space-y-4 rounded-md bg-white/5 px-3 py-3 text-xs">
        {/* 1. Timing */}
        <TimingBar timings={d.timings} />

        {/* 2. Score distribution */}
        {d.score_distribution?.n > 0 && (
          <ScoreHistogram dist={d.score_distribution} />
        )}

        {/* 3. Candidate table */}
        {d.candidates.length > 0 && (
          <CandidateTable candidates={d.candidates} />
        )}

        {/* 4. Budget */}
        {d.budget?.budget_tokens > 0 && (
          <BudgetMeter budget={d.budget} />
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
};
