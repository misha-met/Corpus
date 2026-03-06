"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import * as Dialog from "@radix-ui/react-dialog";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  Cell,
} from "recharts";
import { sourceApi, type AnalyticsResponse } from "@/lib/api-client";

const RelationshipGraphComponent = dynamic(
  () =>
    import("@/components/relationship-graph").then((m) => ({
      default: m.RelationshipGraph,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-64 text-xs text-gray-500">
        Loading graph…
      </div>
    ),
  },
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

const ENTITY_TYPE_COLORS: Record<string, string> = {
  PERSON: "#60a5fa",
  ORG: "#818cf8",
  GPE: "#34d399",
  LOC: "#a3e635",
  NORP: "#fb923c",
  EVENT: "#f472b6",
  WORK_OF_ART: "#e879f9",
  FAC: "#fbbf24",
};

function entityColor(type: string): string {
  return ENTITY_TYPE_COLORS[type] ?? "#94a3b8";
}

const CLUSTER_COLORS = [
  "#60a5fa", "#818cf8", "#34d399", "#a3e635",
  "#fb923c", "#f472b6", "#e879f9", "#fbbf24",
  "#38bdf8", "#a78bfa",
];

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------

function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <div className={`animate-pulse rounded-lg bg-[#1a1a1a] ${className}`} />
  );
}

function SkeletonSection() {
  return (
    <div className="space-y-3">
      <SkeletonCard className="h-5 w-40" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <SkeletonCard key={i} className="h-20" />
        ))}
      </div>
      <SkeletonCard className="h-40" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-[#2a2a2a] bg-[#141414] px-4 py-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-semibold text-gray-100 tabular-nums">{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview section
// ---------------------------------------------------------------------------

function OverviewSection({ data }: { data: AnalyticsResponse["overview"] }) {
  return (
    <section>
      <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Corpus Overview
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Documents" value={data.source_count} />
        <StatCard label="Chunks" value={data.child_chunk_count.toLocaleString()} />
        <StatCard label="Est. Tokens" value={formatTokens(data.estimated_tokens)} />
        <StatCard label="Avg Chunks / Doc" value={data.avg_chunks_per_doc} />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Topic clusters section
// ---------------------------------------------------------------------------

function TopicsSection({ topics }: { topics: AnalyticsResponse["topics"] }) {
  if (topics.length === 0) {
    return (
      <section>
        <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Topic Clusters
        </h2>
        <p className="text-xs text-gray-500">Not enough documents to cluster (need ≥ 2).</p>
      </section>
    );
  }

  const chartData = topics.map((t, i) => ({
    name: t.label.length > 16 ? t.label.slice(0, 16) + "…" : t.label,
    size: t.size,
    color: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
  }));

  return (
    <section>
      <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Topic Clusters
      </h2>
      <div className="mb-4 h-36">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#181818", border: "1px solid #2a2a2a", borderRadius: 6 }}
              labelStyle={{ color: "#e5e7eb", fontSize: 11 }}
              itemStyle={{ color: "#9ca3af", fontSize: 11 }}
            />
            <Bar dataKey="size" radius={[3, 3, 0, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {topics.map((cluster, i) => (
          <div
            key={cluster.cluster_id}
            className="rounded-lg border border-[#2a2a2a] bg-[#141414] p-3"
          >
            <div className="flex items-center gap-2 mb-2">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
              />
              <span className="text-xs font-medium text-gray-200 capitalize">{cluster.label}</span>
              <span className="ml-auto text-xs text-gray-500">{cluster.size} doc{cluster.size !== 1 ? "s" : ""}</span>
            </div>
            <div className="flex flex-wrap gap-1 mb-2">
              {cluster.keywords.slice(0, 6).map((kw) => (
                <span
                  key={kw}
                  className="inline-block rounded px-1.5 py-0.5 text-[10px] bg-[#1f1f1f] text-gray-400 border border-[#2a2a2a]"
                >
                  {kw}
                </span>
              ))}
            </div>
            <p className="text-[10px] text-gray-500 truncate">
              {cluster.source_ids.join(", ")}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Entity frequency section
// ---------------------------------------------------------------------------

function EntitiesSection({
  entities,
  nerAvailable,
}: {
  entities: AnalyticsResponse["entities"];
  nerAvailable: boolean;
}) {
  if (!nerAvailable) {
    return (
      <section>
        <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Named Entities
        </h2>
        <div className="rounded-lg border border-[#2a2a2a] bg-[#141414] px-4 py-3">
          <p className="text-xs text-gray-500">
            Install spaCy + en_core_web_sm for entity analysis:
          </p>
          <code className="mt-1 block text-[10px] text-gray-400 font-mono">
            pip install spacy &amp;&amp; python -m spacy download en_core_web_sm
          </code>
        </div>
      </section>
    );
  }

  if (entities.length === 0) {
    return (
      <section>
        <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Named Entities
        </h2>
        <p className="text-xs text-gray-500">No entities found in the corpus.</p>
      </section>
    );
  }

  const top20 = entities.slice(0, 20);
  const chartData = top20.map((e) => ({
    name: e.text.length > 20 ? e.text.slice(0, 20) + "…" : e.text,
    count: e.count,
    type: e.type,
    color: entityColor(e.type),
  }));

  return (
    <section>
      <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Named Entities
      </h2>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {Object.entries(ENTITY_TYPE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1 text-[10px] text-gray-500">
            <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
            {type}
          </span>
        ))}
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 4, right: 24, left: 4, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 10, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={110}
              tick={{ fontSize: 10, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{ background: "#181818", border: "1px solid #2a2a2a", borderRadius: 6 }}
              labelStyle={{ color: "#e5e7eb", fontSize: 11 }}
              itemStyle={{ color: "#9ca3af", fontSize: 11 }}
              formatter={(value, _name, props) => [value, props.payload.type]}
            />
            <Bar dataKey="count" radius={[0, 3, 3, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Timeline section
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function TimelineTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const bucket = payload[0]?.payload;
  return (
    <div className="rounded border border-[#2a2a2a] bg-[#181818] px-3 py-2 text-xs">
      <p className="text-gray-300 font-medium mb-1">{label}</p>
      <p className="text-gray-400">Mentions: {payload[0]?.value}</p>
      {bucket?.sources?.length > 0 && (
        <p className="text-gray-500 mt-1 text-[10px]">
          Sources: {bucket.sources.slice(0, 3).join(", ")}
          {bucket.sources.length > 3 ? ` +${bucket.sources.length - 3}` : ""}
        </p>
      )}
    </div>
  );
}

function TimelineSection({
  timeline,
  timelineAvailable,
}: {
  timeline: AnalyticsResponse["timeline"];
  timelineAvailable: boolean;
}) {
  if (!timelineAvailable) {
    return (
      <section>
        <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Temporal Distribution
        </h2>
        <div className="rounded-lg border border-[#2a2a2a] bg-[#141414] px-4 py-3">
          <p className="text-xs text-gray-500">
            Install dateparser for temporal extraction:
          </p>
          <code className="mt-1 block text-[10px] text-gray-400 font-mono">
            pip install dateparser
          </code>
        </div>
      </section>
    );
  }

  if (timeline.length === 0) {
    return (
      <section>
        <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Temporal Distribution
        </h2>
        <p className="text-xs text-gray-500">No temporal references detected in the corpus.</p>
      </section>
    );
  }

  const chartData = timeline.map((b) => ({
    label: b.label,
    count: b.count,
    sources: b.sources,
  }));

  return (
    <section>
      <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Temporal Distribution
      </h2>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <defs>
              <linearGradient id="timelineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
            <Tooltip content={<TimelineTooltip />} />
            <Area
              type="monotone"
              dataKey="count"
              stroke="#60a5fa"
              strokeWidth={1.5}
              fill="url(#timelineGradient)"
              dot={false}
              activeDot={{ r: 4, fill: "#60a5fa" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Relationships section
// ---------------------------------------------------------------------------

function RelationshipsSection({
  relationships,
}: {
  relationships: AnalyticsResponse["relationships"];
}) {
  const hasData = relationships && relationships.nodes.length > 0;
  return (
    <section>
      <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Source Relationships
      </h2>
      {hasData ? (
        <>
          <p className="text-xs text-gray-500 mb-3">
            {relationships.nodes.length} sources ·{" "}
            {relationships.edges.length} connections. Hover nodes to highlight
            neighbours; click to pin.
          </p>
          <RelationshipGraphComponent data={relationships} />
          {/* Edge type legend note */}
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-600">
            <span>
              <span className="font-medium text-blue-400/70">Solid</span> =
              embedding similarity
            </span>
            <span>
              <span className="font-medium text-indigo-400/70">Dashed</span> =
              shared entities
            </span>
            <span>
              <span className="font-medium text-emerald-400/70">Dotted</span> =
              temporal overlap
            </span>
          </div>
        </>
      ) : (
        <div className="rounded-lg border border-[#1e1e1e] bg-[#111] px-4 py-6 text-center">
          <p className="text-xs text-gray-500">
            Relationship graph will appear once analytics are computed (at least
            2 sources required).
          </p>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main dashboard component
// ---------------------------------------------------------------------------

interface AnalyticsDashboardProps {
  open: boolean;
  onClose: () => void;
}

export function AnalyticsDashboard({ open, onClose }: AnalyticsDashboardProps) {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async (force = false) => {
    setLoading(true);
    setError(null);
    try {
      const data = await sourceApi.getAnalytics(force);
      setAnalytics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, []);

  // Load when dialog opens
  useEffect(() => {
    if (open && !analytics) {
      fetchAnalytics();
    }
  }, [open, analytics, fetchAnalytics]);

  const isEmpty = analytics && analytics.overview.source_count === 0;

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content
          className="fixed inset-x-4 top-8 bottom-8 z-50 mx-auto max-w-5xl rounded-xl border border-[#2a2a2a] bg-[#0e0e0e] shadow-2xl flex flex-col focus:outline-none"
          aria-describedby={undefined}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#1e1e1e] shrink-0">
            <div className="flex items-center gap-3">
              <svg
                className="w-4.5 h-4.5 text-blue-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <Dialog.Title className="text-sm font-semibold text-gray-200">
                Corpus Analytics
              </Dialog.Title>
            </div>
            <div className="flex items-center gap-2">
              {!loading && analytics && (
                <button
                  onClick={() => fetchAnalytics(true)}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-gray-500 hover:text-gray-300 hover:bg-[#1e1e1e] rounded-md transition-colors"
                  title="Recompute analytics"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M4 4l4.5 4.5A9 9 0 1020 20" />
                  </svg>
                  Refresh
                </button>
              )}
              <Dialog.Close
                className="p-1.5 text-gray-500 hover:text-gray-200 hover:bg-[#1e1e1e] rounded-md transition-colors"
                aria-label="Close"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </Dialog.Close>
            </div>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-8">
            {loading && (
              <div className="space-y-8">
                <SkeletonSection />
                <SkeletonSection />
              </div>
            )}

            {!loading && error && (
              <div className="rounded-lg border border-red-900/40 bg-red-950/20 px-4 py-3">
                <p className="text-xs text-red-400">{error}</p>
              </div>
            )}

            {!loading && isEmpty && (
              <div className="flex flex-col items-center justify-center h-48 text-center">
                <svg className="w-10 h-10 text-gray-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm text-gray-400">Upload documents first to see analytics.</p>
              </div>
            )}

            {!loading && analytics && !isEmpty && (
              <>
                <OverviewSection data={analytics.overview} />
                <TopicsSection topics={analytics.topics} />
                <EntitiesSection
                  entities={analytics.entities}
                  nerAvailable={analytics.ner_available}
                />
                <TimelineSection
                  timeline={analytics.timeline}
                  timelineAvailable={analytics.timeline_available}
                />
                <RelationshipsSection relationships={analytics.relationships} />
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
