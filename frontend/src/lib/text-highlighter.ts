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
  const step = Math.max(10, Math.floor(n.length / 100));
  for (let len = n.length - 1; len >= minLen; len -= step) {
    const sub = n.slice(0, len);
    const idx = h.indexOf(sub);
    if (idx !== -1) {
      let end = idx + sub.length;
      while (end < h.length && h[end] !== " " && (end - (idx + sub.length)) < 20) {
        end++;
      }
      return { start: idx, length: end - idx };
    }
  }

  // Try progressively shorter suffixes
  for (let len = n.length - 1; len >= minLen; len -= step) {
    const sub = n.slice(n.length - len);
    const idx = h.indexOf(sub);
    if (idx !== -1) {
      let end = idx + sub.length;
      while (end < h.length && h[end] !== " " && (end - (idx + sub.length)) < 20) {
        end++;
      }
      return { start: idx, length: end - idx };
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

/**
 * High-level helper: find `chunkText` in the container's text and
 * wrap it with `<mark>`.  Returns the first <mark> element (for
 * scrollIntoView) or null if no match.
 */
export function findAndHighlight(
  container: HTMLElement,
  chunkText: string
): HTMLElement | null {
  clearHighlights(container);
  const rawText = container.innerText ?? container.textContent ?? "";
  const result = findBestMatch(rawText, chunkText);
  if (!result) return null;

  // Map normalised offsets back to raw text offsets.
  // Build a mapping: for each index in the normalised string,
  // track the corresponding index in the raw string.
  const raw = rawText;
  const normToRaw: number[] = [];
  let ri = 0;
  let inWhitespace = false;

  // Skip leading whitespace (mirrors the .trim() in normalise)
  let rawStart = 0;
  while (rawStart < raw.length && /\s/.test(raw[rawStart])) {
    rawStart++;
  }
  ri = rawStart;

  for (; ri < raw.length; ri++) {
    const ch = raw[ri];
    if (/\s/.test(ch)) {
      if (!inWhitespace) {
        // This whitespace char becomes the single space in normalised text
        normToRaw.push(ri);
        inWhitespace = true;
      }
      // Additional whitespace chars are collapsed — no normalised index
    } else {
      normToRaw.push(ri);
      inWhitespace = false;
    }
  }

  const rawMatchStart = normToRaw[result.start] ?? 0;
  const rawMatchEnd =
    normToRaw[result.start + result.length - 1] ?? rawMatchStart;
  // +1 because highlightRange end is exclusive
  const rawLength = rawMatchEnd - rawMatchStart + 1;

  highlightRange(container, rawMatchStart, rawLength);

  return container.querySelector("mark.citation-highlight");
}
