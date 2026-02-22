/**
 * format-citations.ts
 *
 * Formats the cited (green) citations for copy-to-clipboard.
 *
 * Source of truth for the reference label, in priority order:
 *   1. User-provided citation reference saved at ingest time (localStorage)
 *   2. Fallback: the source_id (filename stem) with underscores → spaces
 *
 * No attempt is made to parse or guess author/year from filenames.
 * Users provide a canonical reference string at ingest time via the
 * "Citation reference" optional field in the ingest modal.
 */

import type { Citation } from "./event-parser";
import { getCitationMeta } from "./citation-meta-store";

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/**
 * Returns the display label for a source:
 *   - The user-provided citation reference (if saved at ingest), or
 *   - The source_id cleaned up: underscores/hyphens → spaces, title-cased.
 */
function sourceLabel(sourceId: string): string {
  const stored = getCitationMeta(sourceId);
  if (stored) return stored;
  return sourceId
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Footnote style
// ---------------------------------------------------------------------------

/**
 * Academic footnote format — one numbered entry per cited chunk, in the
 * order they appear in the answer.
 *
 * With citation reference saved:
 *   1. Smith, J. et al. (2024) 'Climate Change Review', p.16.
 *
 * Fallback (filename only):
 *   1. Battery Storage Review, p.4.
 */
export function formatFootnotes(citedCitations: Citation[]): string {
  return citedCitations
    .map((c, i) => {
      const label = sourceLabel(c.source_id);
      const page = c.page != null ? `, p.${c.page}` : "";
      return `${i + 1}. ${label}${page}.`;
    })
    .join("\n");
}

// ---------------------------------------------------------------------------
// Harvard bibliography style
// ---------------------------------------------------------------------------

/**
 * Harvard bibliography format — one entry per source document, with all
 * cited page numbers collapsed together.
 *
 * With citation reference saved:
 *   Smith, J. et al. (2024) 'Climate Change Review'. pp. 8, 16.
 *
 * Fallback (filename only):
 *   Battery Storage Review. p.4.
 */
export function formatHarvardBibliography(citedCitations: Citation[]): string {
  // Group by source_id, preserving first-seen order
  const sourceOrder: string[] = [];
  const bySource = new Map<string, number[]>();

  for (const c of citedCitations) {
    if (!bySource.has(c.source_id)) {
      sourceOrder.push(c.source_id);
      bySource.set(c.source_id, []);
    }
    if (c.page != null) {
      const pages = bySource.get(c.source_id)!;
      if (!pages.includes(c.page)) pages.push(c.page);
    }
  }

  return sourceOrder
    .map((sid) => {
      const label = sourceLabel(sid);
      const pages = (bySource.get(sid) ?? []).sort((a, b) => a - b);
      const pagesStr =
        pages.length === 0
          ? ""
          : pages.length === 1
          ? `. p. ${pages[0]}.`
          : `. pp. ${pages.join(", ")}.`;
      return `${label}${pagesStr || "."}`;
    })
    .join("\n");
}
