"use client";

import { useEffect, useRef, useState } from "react";

export interface IndexedStats {
  sourceCount: number;
  estimatedTokens: number;
}

const POLL_INTERVAL_MS = 8_000;

async function fetchStats(): Promise<IndexedStats> {
  const res = await fetch("/api/sources");
  if (!res.ok) throw new Error(`/api/sources returned ${res.status}`);
  const data = (await res.json()) as { sources: { content_size_bytes?: number | null }[] };
  const sources = data.sources ?? [];
  const totalBytes = sources.reduce(
    (sum, s) => sum + (s.content_size_bytes ?? 0),
    0,
  );
  return {
    sourceCount: sources.length,
    // ~4 bytes per token is a reliable rough estimate for English text
    estimatedTokens: Math.round(totalBytes / 4),
  };
}

/**
 * Polls /api/sources and returns live source count + estimated token count.
 * Refreshes every POLL_INTERVAL_MS so the welcome screen updates after ingest.
 */
export function useIndexedStats(): IndexedStats {
  const [stats, setStats] = useState<IndexedStats>({ sourceCount: 0, estimatedTokens: 0 });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = () => {
      fetchStats()
        .then((s) => { if (!cancelled) setStats(s); })
        .catch(() => {/* silently ignore network errors */})
        .finally(() => {
          if (!cancelled) {
            timerRef.current = setTimeout(load, POLL_INTERVAL_MS);
          }
        });
    };

    load();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return stats;
}
