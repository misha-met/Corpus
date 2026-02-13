/**
 * Text matching and DOM highlighting utilities for citation viewer.
 *
 * Provides fuzzy substring matching against a document and DOM-level
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

/**
 * Find the best substring match of `needle` inside `haystack`.
 *
 * Tries exact match first, then progressively shorter prefixes of the
 * needle (down to 50 characters).  Returns null if no match is found.
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

  // Try progressively shorter prefixes
  const minLen = Math.min(50, n.length);
  for (let len = Math.min(200, n.length); len >= minLen; len -= 10) {
    const sub = n.slice(0, len);
    const idx = h.indexOf(sub);
    if (idx !== -1) {
      return { start: idx, length: sub.length };
    }
  }

  // Try progressively shorter suffixes
  for (let len = Math.min(200, n.length); len >= minLen; len -= 10) {
    const sub = n.slice(n.length - len);
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

/**
 * High-level helper: find `chunkText` in the container's text and
 * wrap it with `<mark>`.  Returns the first <mark> element (for
 * scrollIntoView) or null if no match.
 */
export function findAndHighlight(
  container: HTMLElement,
  chunkText: string
): HTMLElement | null {
  const flatText = container.innerText ?? container.textContent ?? "";
  const result = findBestMatch(flatText, chunkText);
  if (!result) return null;

  highlightRange(container, result.start, result.length);

  return container.querySelector("mark.citation-highlight");
}
