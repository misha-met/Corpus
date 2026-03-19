/**
 * format-citations.ts
 *
 * Formats the cited (green) citations for copy-to-clipboard.
 *
 * Source of truth for the reference label, in priority order:
 *   1. User-provided citation reference persisted in backend source metadata
 *   2. Fallback: the source_id (filename stem) with underscores → spaces
 *
 * No attempt is made to parse or guess author/year from filenames.
 * Users provide a canonical reference string at ingest time via the
 * "Citation reference" optional field in the ingest modal.
 */

import type { Citation } from "./event-parser";

const INLINE_CITATION_PATTERN = "\\[(\\d+)(?:\\s*,\\s*p\\.?\\s*(\\d+))?\\]";

export type InlineCitationMarker = {
  number: number;
  page: number | null;
};

export function extractInlineCitationMarkers(messageText: string): InlineCitationMarker[] {
  const markers: InlineCitationMarker[] = [];
  if (!messageText) return markers;

  const regex = new RegExp(INLINE_CITATION_PATTERN, "gi");
  for (const match of messageText.matchAll(regex)) {
    const number = Number.parseInt(match[1], 10);
    if (!Number.isInteger(number) || number < 1) continue;

    const pageRaw = match[2];
    const parsedPage = pageRaw != null ? Number.parseInt(pageRaw, 10) : Number.NaN;
    const page = Number.isInteger(parsedPage) && parsedPage >= 1 ? parsedPage : null;
    markers.push({ number, page });
  }

  return markers;
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/**
 * Returns the display label for a source:
 *   - The user-provided citation reference (if saved at ingest), or
 *   - The source_id cleaned up: underscores/hyphens → spaces, title-cased.
 */
function sourceLabel(sourceId: string, citationReferenceBySource?: Map<string, string>): string {
  const stored = citationReferenceBySource?.get(sourceId);
  if (stored) {
    return stored;
  }
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
export function formatFootnotes(
  citedCitations: Citation[],
  citationReferenceBySource?: Map<string, string>
): string {
  return citedCitations
    .map((c, i) => {
      const label = sourceLabel(c.source_id, citationReferenceBySource);
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
export function formatHarvardBibliography(
  citedCitations: Citation[],
  citationReferenceBySource?: Map<string, string>
): string {
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
      const label = sourceLabel(sid, citationReferenceBySource);
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

// ---------------------------------------------------------------------------
// Footnote style with renumbered in-text markers
// ---------------------------------------------------------------------------

/**
 * Produces renumbered body text + a sequential footnote list.
 *
 * The [N] markers in the message text may be sparse (e.g. [1], [3], [5]).
 * This function:
 *   1. Scans messageText for all [N] patterns in first-appearance order.
 *   2. Builds a renumber map  {originalN → newSequentialN}.
 *   3. Replaces every [N] in the text with the renumbered equivalent.
 *   4. Builds a footnote list with the new sequential numbers, using
 *      source/page data from citedCitations looked up by original number.
 *
 * clipboard output:
 *   <renumbered body text>
 *
 *   1. Smith, J. et al. (2024) 'Climate Change Review', p.16.
 *   2. Battery Storage Review, p.4.
 */
export function formatFootnotesWithText(
  messageText: string,
  citedCitations: Citation[],
  citationReferenceBySource?: Map<string, string>
): string {
  // 1. Collect original citation numbers in first-appearance order.
  const markers = extractInlineCitationMarkers(messageText);
  const seen = new Set<number>();
  const ordered: number[] = [];
  const markerPageByOriginal = new Map<number, number>();
  for (const marker of markers) {
    const n = marker.number;
    if (!seen.has(n)) {
      seen.add(n);
      ordered.push(n);
    }
    if (marker.page != null && !markerPageByOriginal.has(n)) {
      markerPageByOriginal.set(n, marker.page);
    }
  }

  // 2. Build renumber map: originalN → newSequential (1-based)
  const renumberMap = new Map<number, number>();
  ordered.forEach((orig, idx) => renumberMap.set(orig, idx + 1));

  // 3. Replace [N] and [N, p.XX] in body text with renumbered equivalents.
  const renumberedText = messageText.replace(new RegExp(INLINE_CITATION_PATTERN, "gi"), (_, digits, pageDigits) => {
    const orig = parseInt(digits, 10);
    const newN = renumberMap.get(orig);
    if (newN === undefined) {
      return pageDigits ? `[${digits}, p.${pageDigits}]` : `[${digits}]`;
    }
    return pageDigits ? `[${newN}, p.${pageDigits}]` : `[${newN}]`;
  });

  // 4. Build footnote list — look up each original number in citedCitations
  //    (citedCitations is indexed by citation.number which equals the original N)
  const citationsByOriginal = new Map<number, Citation>();
  for (const c of citedCitations) {
    citationsByOriginal.set(c.number, c);
  }

  const footnotes = ordered.map((orig, idx) => {
    const newN = idx + 1;
    const c = citationsByOriginal.get(orig);
    if (!c) return `${newN}. [${orig}].`;
    const label = sourceLabel(c.source_id, citationReferenceBySource);
    const markerPage = markerPageByOriginal.get(orig);
    const pageValue = markerPage ?? c.page ?? null;
    const page = pageValue != null ? `, p.${pageValue}` : "";
    return `${newN}. ${label}${page}.`;
  });

  if (footnotes.length === 0) return renumberedText;
  return `${renumberedText}\n\n${footnotes.join("\n")}`;
}
