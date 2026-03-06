"use client";

/**
 * RelationshipGraph — force-directed graph of source relationships.
 *
 * Loaded client-side only (canvas API).  Consumed by AnalyticsDashboard via
 * next/dynamic with ssr:false.
 *
 * Edge encoding:
 *   solid line      → similarity
 *   dashed [4,4]    → entities
 *   dotted [2,6]    → temporal
 *   mixed types     → solid (takes the highest-priority encoding present)
 *
 * Opacity of non-highlighted nodes/edges is reduced to 20% when a node is
 * hovered or pinned.
 */

import { useRef, useState, useMemo, useCallback } from "react";
import dynamic from "next/dynamic";

import type { RelationshipGraph as RelationshipGraphData } from "@/lib/api-client";

// ── ForceGraph2D: client-only dynamic import ──────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ForceGraph2D = dynamic(() => import("react-force-graph-2d") as any, {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-xs text-gray-500">
      Loading graph…
    </div>
  ),
// eslint-disable-next-line @typescript-eslint/no-explicit-any
}) as any;

// ── Colour palette ────────────────────────────────────────────────────────

const CLUSTER_COLORS = [
  "#60a5fa", // blue-400
  "#818cf8", // indigo-400
  "#34d399", // emerald-400
  "#a3e635", // lime-400
  "#fb923c", // orange-400
  "#f472b6", // pink-400
  "#e879f9", // fuchsia-400
  "#fbbf24", // amber-400
  "#38bdf8", // sky-400
  "#a78bfa", // violet-400
];

function topicColor(topicId: number | null | undefined): string {
  if (topicId == null) return "#94a3b8"; // slate-400 fallback
  return CLUSTER_COLORS[topicId % CLUSTER_COLORS.length];
}

/** Append 2-char hex alpha suffix to a 6-char hex color string. */
function withAlpha(hex: string, alpha: number): string {
  const a = Math.round(alpha * 255)
    .toString(16)
    .padStart(2, "0");
  return hex + a;
}

// ── Component ─────────────────────────────────────────────────────────────

interface Props {
  data: RelationshipGraphData;
}

interface GraphNode {
  id: string;
  label: string;
  size: number;
  dominant_topic: number | null;
  summary: string | null;
  // d3-force augmented fields (mutable at runtime)
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  _types: string[];
  _combinedWeight: number;
}

export function RelationshipGraph({ data }: Props) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);
  const engineStoppedRef = useRef(false);

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeId, setPinnedNodeId] = useState<string | null>(null);

  // Filter controls
  const [minWeight, setMinWeight] = useState(0.1);
  const [showSimilarity, setShowSimilarity] = useState(true);
  const [showEntities, setShowEntities] = useState(true);
  const [showTemporal, setShowTemporal] = useState(true);

  const activeNodeId = hoveredNodeId ?? pinnedNodeId;

  // ── Build neighbor lookup from full edge set ───────────────────────────
  const neighborMap = useMemo<Record<string, Set<string>>>(() => {
    const map: Record<string, Set<string>> = {};
    for (const n of data.nodes) {
      map[n.id] = new Set();
    }
    for (const e of data.edges) {
      map[e.source]?.add(e.target);
      map[e.target]?.add(e.source);
    }
    return map;
  }, [data]);

  // ── Active neighbor set (changes on hover / pin) ───────────────────────
  const activeNeighbors = useMemo<Set<string>>(() => {
    if (!activeNodeId) return new Set();
    const s = new Set<string>([activeNodeId]);
    neighborMap[activeNodeId]?.forEach((n) => s.add(n));
    return s;
  }, [activeNodeId, neighborMap]);

  // ── Filtered graph data passed to ForceGraph2D ─────────────────────────
  const graphData = useMemo(() => {
    const links: GraphLink[] = data.edges
      .filter((e) => e.combined_weight >= minWeight)
      .filter((e) => {
        // Hide links whose *only* type is unchecked
        const onlySim = e.types.every((t) => t === "similarity");
        const onlyEnt = e.types.every((t) => t === "entities");
        const onlyTmp = e.types.every((t) => t === "temporal");
        if (!showSimilarity && onlySim) return false;
        if (!showEntities && onlyEnt) return false;
        if (!showTemporal && onlyTmp) return false;
        return true;
      })
      .map((e) => ({
        source: e.source,
        target: e.target,
        _types: e.types,
        _combinedWeight: e.combined_weight,
      }));

    const nodes: GraphNode[] = data.nodes.map((n) => ({
      id: n.id,
      label: n.label,
      size: n.size,
      dominant_topic: n.dominant_topic,
      summary: n.summary,
    }));

    return { nodes, links };
  }, [data, minWeight, showSimilarity, showEntities, showTemporal]);

  // ── Visual callbacks ───────────────────────────────────────────────────

  const nodeColor = useCallback(
    (node: GraphNode) => {
      const base = topicColor(node.dominant_topic);
      if (activeNodeId && !activeNeighbors.has(node.id)) {
        return withAlpha(base, 0.18);
      }
      return base;
    },
    [activeNodeId, activeNeighbors],
  );

  const nodeVal = useCallback((node: GraphNode) => {
    return Math.max(2, Math.log(node.size + 1) * 4);
  }, []);

  const linkColor = useCallback(
    (link: GraphLink) => {
      const types = link._types;
      let base = "#4b5563"; // gray-600 default
      if (types.includes("similarity")) base = "#60a5fa";
      else if (types.includes("entities")) base = "#818cf8";
      else if (types.includes("temporal")) base = "#34d399";
      const srcId = typeof link.source === "string" ? link.source : link.source.id;
      const tgtId = typeof link.target === "string" ? link.target : link.target.id;
      if (
        activeNodeId &&
        !activeNeighbors.has(srcId) &&
        !activeNeighbors.has(tgtId)
      ) {
        return withAlpha(base, 0.12);
      }
      return withAlpha(base, 0.6);
    },
    [activeNodeId, activeNeighbors],
  );

  const linkWidth = useCallback((link: GraphLink) => {
    return 0.5 + link._combinedWeight * 2;
  }, []);

  const linkLineDash = useCallback((link: GraphLink) => {
    const types = link._types;
    if (!types.includes("similarity") && types.includes("entities")) return [4, 4];
    if (types.length === 1 && types[0] === "temporal") return [2, 6];
    return null;
  }, []);

  // ── Interaction handlers ───────────────────────────────────────────────

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNodeId(node ? node.id : null);
  }, []);

  const handleNodeClick = useCallback((node: GraphNode) => {
    setPinnedNodeId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const handleBackgroundClick = useCallback(() => {
    setPinnedNodeId(null);
  }, []);

  const handleRecenter = useCallback(() => {
    fgRef.current?.zoomToFit(400);
  }, []);

  // ── Selected node sidebar ──────────────────────────────────────────────
  const selectedNode = pinnedNodeId
    ? data.nodes.find((n) => n.id === pinnedNodeId) ?? null
    : null;

  return (
    <div className="relative rounded-lg overflow-hidden border border-[#1e1e1e]" style={{ height: 520 }}>
      {/* Force graph canvas */}
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        backgroundColor="#0a0a0a"
        nodeId="id"
        nodeLabel={(node: GraphNode) => `${node.label} · ${node.size} chunks`}
        nodeColor={nodeColor}
        nodeVal={nodeVal}
        linkSource="source"
        linkTarget="target"
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkLineDash={linkLineDash}
        onNodeHover={handleNodeHover}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        cooldownTicks={100}
        warmupTicks={50}
        onEngineStop={() => {
          if (!engineStoppedRef.current && fgRef.current) {
            fgRef.current.zoomToFit(400);
            engineStoppedRef.current = true;
          }
        }}
      />

      {/* ── Filter controls ──────────────────────────────────────────── */}
      <div className="absolute top-2 left-2 flex flex-col gap-1.5 bg-[#111]/85 backdrop-blur-sm border border-[#2a2a2a] rounded-lg px-3 py-2 text-xs text-gray-400 min-w-40">
        <div className="flex items-center justify-between gap-3">
          <label className="text-gray-500 shrink-0">Min weight</label>
          <span className="tabular-nums text-gray-300">{minWeight.toFixed(1)}</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={0.8}
          step={0.05}
          value={minWeight}
          onChange={(e) => setMinWeight(Number(e.target.value))}
          className="w-full h-1 accent-blue-400 cursor-pointer"
        />
        <div className="flex flex-col gap-1 mt-0.5">
          {[
            { key: "similarity", label: "Similarity", state: showSimilarity, set: setShowSimilarity, color: "#60a5fa" },
            { key: "entities", label: "Entities", state: showEntities, set: setShowEntities, color: "#818cf8" },
            { key: "temporal", label: "Temporal", state: showTemporal, set: setShowTemporal, color: "#34d399" },
          ].map(({ key, label, state, set, color }) => (
            <label key={key} className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={state}
                onChange={(e) => set(e.target.checked)}
                className="accent-blue-400 w-3 h-3"
              />
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{ background: color }}
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
        <button
          onClick={handleRecenter}
          className="mt-1 text-xs text-gray-500 hover:text-gray-200 hover:bg-[#1e1e1e] rounded px-2 py-1 transition-colors text-center"
        >
          ⟳ Recenter
        </button>
      </div>

      {/* ── Legend ────────────────────────────────────────────────────── */}
      <div className="absolute bottom-2 left-2 flex flex-col gap-1 bg-[#111]/85 border border-[#2a2a2a] rounded-lg px-2.5 py-1.5 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="#60a5fa" strokeWidth="1.5" /></svg>
          <span>Similarity</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="#818cf8" strokeWidth="1.5" strokeDasharray="4 4" /></svg>
          <span>Entities</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="#34d399" strokeWidth="1.5" strokeDasharray="2 6" /></svg>
          <span>Temporal</span>
        </div>
      </div>

      {/* ── Selected node panel ───────────────────────────────────────── */}
      {selectedNode && (
        <div className="absolute top-2 right-2 w-52 bg-[#111]/90 backdrop-blur-sm border border-[#2a2a2a] rounded-lg px-3 py-2.5 text-xs text-gray-300 space-y-1.5">
          <div className="flex items-start justify-between gap-2">
            <p
              className="font-semibold text-gray-100 leading-tight break-all"
              style={{ color: topicColor(selectedNode.dominant_topic) }}
            >
              {selectedNode.label}
            </p>
            <button
              onClick={() => setPinnedNodeId(null)}
              className="text-gray-600 hover:text-gray-300 shrink-0"
              aria-label="Deselect"
            >
              ✕
            </button>
          </div>
          <p className="text-gray-500">{selectedNode.size.toLocaleString()} chunks</p>
          {selectedNode.dominant_topic != null && (
            <p className="text-gray-500">Topic cluster {selectedNode.dominant_topic}</p>
          )}
          {selectedNode.summary && (
            <p className="text-gray-400 leading-relaxed line-clamp-4">{selectedNode.summary}</p>
          )}
          {/* Neighbours list */}
          {neighborMap[selectedNode.id] && neighborMap[selectedNode.id].size > 0 && (
            <div>
              <p className="text-gray-600 mb-0.5">Connected to:</p>
              <ul className="space-y-0.5">
                {[...neighborMap[selectedNode.id]].slice(0, 6).map((nid) => (
                  <li
                    key={nid}
                    className="truncate text-gray-500 hover:text-gray-300 cursor-pointer"
                    onClick={() => setPinnedNodeId(nid)}
                  >
                    {nid}
                  </li>
                ))}
                {neighborMap[selectedNode.id].size > 6 && (
                  <li className="text-gray-600">
                    +{neighborMap[selectedNode.id].size - 6} more
                  </li>
                )}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {data.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-xs text-gray-500">No relationship data available yet.</p>
        </div>
      )}
    </div>
  );
}
