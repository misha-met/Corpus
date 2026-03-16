/**
 * Text matching and DOM highlighting utilities for citation viewer.
 *
 * Provides exact substring matching against a document and DOM-level
 * wrapping of matched text in <mark> elements.
 */

/** Normalise text for comparison: collapse whitespace, trim, lowercase. */
function normalise(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

/** Result of a fuzzy text search. */
export interface MatchResult {
  /** Start character index in the normalised document text. */
  start: number;
  /** Length of the matched substring. */
  length: number;
}

interface RawMatchResult {
  /** Start character index in the raw document text. */
  start: number;
  /** Raw character length of the matched substring. */
  length: number;
}

/**
 * Find an exact (normalised) substring match of `needle` inside `haystack`,
 * with a progressive prefix fallback for robustness against minor whitespace
 * or encoding differences between stored chunk text and rendered document text.
 *
 * Returns null if no match is found.
 */
export function findBestMatch(
  haystack: string,
  needle: string
): MatchResult | null {
  const h = normalise(haystack);
  const n = normalise(needle);

  if (!n || !h) return null;

  // Exact full match
  const exactIdx = h.indexOf(n);
  if (exactIdx !== -1) {
    return { start: exactIdx, length: n.length };
  }

  // Prefix fallback: try progressively shorter prefixes down to 80 chars.
  // This handles minor whitespace/encoding differences at chunk boundaries.
  const minLen = Math.max(80, Math.floor(n.length * 0.4));
  const step = Math.max(20, Math.floor(n.length / 20));
  for (let len = n.length - step; len >= minLen; len -= step) {
    const sub = n.slice(0, len);
    const idx = h.indexOf(sub);
    if (idx !== -1) {
      return { start: idx, length: sub.length };
    }
  }

  return null;
}

/**
 * Walk text nodes inside a container and wrap the range that corresponds
 * to the character offsets [start, start+length) of the container's
 * flat text content in `<mark class="citation-highlight">` elements.
 *
 * Returns true if any wrapping was performed.
 */
export function highlightRange(
  container: HTMLElement,
  start: number,
  length: number
): boolean {
  if (length <= 0) return false;

  const textNodes: Text[] = [];
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    textNodes.push(node);
  }

  let charOffset = 0;
  const end = start + length;
  let wrapped = false;

  for (const tNode of textNodes) {
    const nodeText = tNode.textContent ?? "";
    const nodeStart = charOffset;
    const nodeEnd = charOffset + nodeText.length;
    charOffset = nodeEnd;

    // No overlap
    if (nodeEnd <= start || nodeStart >= end) continue;

    // Calculate overlap within this text node
    const overlapStart = Math.max(start, nodeStart) - nodeStart;
    const overlapEnd = Math.min(end, nodeEnd) - nodeStart;

    if (overlapStart >= overlapEnd) continue;

    // Split the text node and wrap the matching portion
    const before = nodeText.slice(0, overlapStart);
    const match = nodeText.slice(overlapStart, overlapEnd);
    const after = nodeText.slice(overlapEnd);

    const parent = tNode.parentNode;
    if (!parent) continue;

    const mark = document.createElement("mark");
    mark.className = "citation-highlight";
    mark.textContent = match;

    const frag = document.createDocumentFragment();
    if (before) frag.appendChild(document.createTextNode(before));
    frag.appendChild(mark);
    if (after) frag.appendChild(document.createTextNode(after));

    parent.replaceChild(frag, tNode);
    wrapped = true;
  }

  return wrapped;
}

export function clearHighlights(container: HTMLElement): void {
  const marks = container.querySelectorAll("mark.citation-highlight");
  marks.forEach((mark) => {
    const parent = mark.parentNode;
    if (!parent) return;
    // Replace <mark> with its text content
    const text = document.createTextNode(mark.textContent ?? "");
    parent.replaceChild(text, mark);
    // Merge adjacent text nodes
    parent.normalize();
  });
}

/** Build a normalised-index → raw-index mapping for a raw string.
 * Mirrors `normalise()`: collapses whitespace runs to one space, trims leading whitespace. */
function buildNormToRaw(raw: string): number[] {
  const normToRaw: number[] = [];
  let inWhitespace = false;
  let rawStart = 0;
  while (rawStart < raw.length && /\s/.test(raw[rawStart])) {
    rawStart++;
  }
  for (let ri = rawStart; ri < raw.length; ri++) {
    const ch = raw[ri];
    if (/\s/.test(ch)) {
      if (!inWhitespace) {
        normToRaw.push(ri);
        inWhitespace = true;
      }
    } else {
      normToRaw.push(ri);
      inWhitespace = false;
    }
  }
  return normToRaw;
}

/** Convert a normalised match into raw-text indices. */
function findRawMatch(rawText: string, needle: string): RawMatchResult | null {
  const result = findBestMatch(rawText, needle);
  if (!result) return null;

  const normToRaw = buildNormToRaw(rawText);
  if (normToRaw.length === 0) return null;

  const rawStart = normToRaw[result.start] ?? 0;
  const rawEnd = normToRaw[result.start + result.length - 1] ?? rawStart;
  return {
    start: rawStart,
    length: Math.max(1, rawEnd - rawStart + 1),
  };
}

/**
 * Find `chunkText` in the container's text, wrap it with `<mark>`, and
 * return the element to scroll to.
 *
 * When `scrollToText` is provided (e.g. the child chunk inside a highlighted
 * parent chunk), the function highlights `chunkText` in full but returns the
 * `<mark>` element that covers the start of `scrollToText`.  This lets the
 * caller scroll to the relevant citation position rather than the top of the
 * (potentially larger) highlighted region.
 */
export function findAndHighlight(
  container: HTMLElement,
  chunkText: string,
  scrollToText?: string,
  scopeText?: string,
): HTMLElement | null {
  clearHighlights(container);
  const rawText = container.innerText ?? container.textContent ?? "";

  let scopeRawMatch: RawMatchResult | null = null;
  let targetRawMatch: RawMatchResult | null = null;

  if (scopeText) {
    scopeRawMatch = findRawMatch(rawText, scopeText);
    if (scopeRawMatch) {
      const scopeSlice = rawText.slice(
        scopeRawMatch.start,
        scopeRawMatch.start + scopeRawMatch.length,
      );
      const localTargetMatch = findRawMatch(scopeSlice, chunkText);
      if (localTargetMatch) {
        targetRawMatch = {
          start: scopeRawMatch.start + localTargetMatch.start,
          length: localTargetMatch.length,
        };
      }
    }
  }

  if (!targetRawMatch) {
    targetRawMatch = findRawMatch(rawText, chunkText);
  }
  if (!targetRawMatch) return null;

  const rawMatchStart = targetRawMatch.start;
  const rawLength = targetRawMatch.length;

  // Compute the raw offset of the scroll target BEFORE mutating the DOM.
  let scrollRawOffset: number | null = null;
  if (scrollToText) {
    if (scopeRawMatch) {
      const scopeSlice = rawText.slice(
        scopeRawMatch.start,
        scopeRawMatch.start + scopeRawMatch.length,
      );
      const localScrollMatch = findRawMatch(scopeSlice, scrollToText);
      if (localScrollMatch) {
        scrollRawOffset = scopeRawMatch.start + localScrollMatch.start;
      }
    }
    if (scrollRawOffset === null) {
      const globalScrollMatch = findRawMatch(rawText, scrollToText);
      if (globalScrollMatch) {
        scrollRawOffset = globalScrollMatch.start;
      }
    }
  }

  highlightRange(container, rawMatchStart, rawLength);

  // If a scroll target was given, find the <mark> element that covers it.
  // After highlightRange the marks appear in document order and their
  // cumulative textContent mirrors the raw character range they span.
  if (scrollRawOffset !== null) {
    const marks = Array.from(
      container.querySelectorAll("mark.citation-highlight")
    ) as HTMLElement[];
    let cursor = rawMatchStart;
    for (const mark of marks) {
      const markLen = (mark.textContent ?? "").length;
      if (scrollRawOffset < cursor + markLen) {
        return mark;
      }
      cursor += markLen;
    }
  }

  return container.querySelector("mark.citation-highlight") as HTMLElement | null;
}
