"""Post-hoc citation highlight verification.

When context expansion is active, the generator sees the full parent chunk
(~1000-1500 tokens) but citations are anchored to the child chunk (~200-300
tokens).  If the model cites content from the parent that falls outside the
child chunk boundaries, the frontend highlights the wrong region.

This module provides a lightweight verification pass that runs after
generation completes.  For each citation whose referenced content falls
outside the child chunk, it returns a ``highlight_text`` passage extracted
from the parent chunk so the frontend can highlight the correct region.
"""

from __future__ import annotations

import re
from typing import Optional

# Words to skip during stem extraction (common English stopwords)
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "its", "let", "may", "who", "did", "get", "him", "his",
    "how", "its", "may", "new", "now", "old", "see", "way",
    "any", "few", "got", "use", "also", "back", "been", "call",
    "come", "each", "find", "from", "give", "have", "here",
    "just", "know", "like", "long", "look", "make", "many",
    "more", "much", "must", "name", "only", "over", "part",
    "some", "such", "take", "than", "that", "them", "then",
    "they", "this", "time", "very", "what", "when", "will",
    "with", "work", "year", "your", "into", "most", "about",
    "being", "could", "after", "again", "other", "their",
    "there", "these", "those", "under", "which", "where",
    "while", "would", "should", "still", "through",
})

# Sentence boundary pattern: period/exclamation/question followed by
# whitespace or end-of-string, or double newline.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|\n\n+')

# Citation markers: [N] or [CHUNK N]
_CITATION_MARKER = re.compile(r'\[(?:CHUNK\s+)?(\d+)\]')


def _extract_claims(answer_text: str) -> dict[int, list[str]]:
    """Parse the answer to extract claim sentences for each citation number.

    Returns ``{citation_number: [sentence1, sentence2, ...]}`` where each
    sentence is the text surrounding a ``[N]`` or ``[CHUNK N]`` marker,
    with citation markers stripped.
    """
    claims: dict[int, list[str]] = {}

    # Split into sentences using boundary pattern
    sentences = _SENTENCE_END.split(answer_text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find all citation markers in this sentence
        for m in _CITATION_MARKER.finditer(sentence):
            cit_num = int(m.group(1))
            # Strip all citation markers from the sentence for clean stems
            clean = _CITATION_MARKER.sub('', sentence).strip()
            if clean:
                claims.setdefault(cit_num, []).append(clean)

    return claims


def _extract_stems(text: str) -> list[str]:
    """Extract 6-char prefix stems from content words.

    Filters out stopwords and words shorter than 4 characters.
    The 6-char prefix approach naturally groups word families
    (e.g. "describe"/"description"/"describing" → "descri").
    """
    words = re.findall(r'[a-zA-Z]+', text.lower())
    stems = []
    for w in words:
        if len(w) < 4:
            continue
        if w in _STOPWORDS:
            continue
        stems.append(w[:6])
    return stems


def _find_stem_positions(stems: list[str], parent_text: str) -> list[tuple[int, str]]:
    """Find character positions where stems match words in parent text.

    Returns ``[(position, stem), ...]`` sorted by position.
    """
    hits: list[tuple[int, str]] = []
    parent_lower = parent_text.lower()
    seen_patterns: dict[str, list[int]] = {}

    for stem in stems:
        if stem in seen_patterns:
            for pos in seen_patterns[stem]:
                hits.append((pos, stem))
            continue

        positions = []
        pattern = re.compile(r'\b' + re.escape(stem) + r'[a-z]*', re.IGNORECASE)
        for m in pattern.finditer(parent_lower):
            positions.append(m.start())
        seen_patterns[stem] = positions
        for pos in positions:
            hits.append((pos, stem))

    hits.sort(key=lambda x: x[0])
    return hits


def _best_density_window(
    hits: list[tuple[int, str]],
    window_size: int = 600,
) -> Optional[tuple[int, int, int]]:
    """Slide a window across hit positions, find max unique-stem density.

    Returns ``(window_start, window_end, unique_stem_count)`` or None if
    fewer than 3 unique stems found in any window.
    """
    if not hits:
        return None

    best_start = 0
    best_end = 0
    best_count = 0

    positions = [h[0] for h in hits]
    stems_at = [h[1] for h in hits]

    # Sliding window using two pointers
    left = 0
    stem_counts: dict[str, int] = {}
    unique_count = 0

    for right in range(len(hits)):
        # Add right element
        s = stems_at[right]
        stem_counts[s] = stem_counts.get(s, 0) + 1
        if stem_counts[s] == 1:
            unique_count += 1

        # Shrink window from left if too wide
        while positions[right] - positions[left] > window_size:
            ls = stems_at[left]
            stem_counts[ls] -= 1
            if stem_counts[ls] == 0:
                unique_count -= 1
                del stem_counts[ls]
            left += 1

        if unique_count > best_count:
            best_count = unique_count
            best_start = positions[left]
            best_end = positions[right]

    if best_count < 3:
        return None

    return (best_start, best_end, best_count)


def _find_child_span(
    child_text: str,
    parent_text: str,
) -> Optional[tuple[int, int]]:
    """Locate the child chunk text within the parent text.

    Tries exact substring match first, then progressively shorter prefixes
    to handle boundary whitespace differences.
    """
    child_stripped = child_text.strip()
    if not child_stripped:
        return None

    # Exact match
    idx = parent_text.find(child_stripped)
    if idx != -1:
        return (idx, idx + len(child_stripped))

    # Try progressively shorter prefixes (handle boundary trimming)
    for trim in range(10, min(60, len(child_stripped)), 10):
        prefix = child_stripped[trim:]
        if len(prefix) < 40:
            break
        idx = parent_text.find(prefix)
        if idx != -1:
            return (idx, idx + len(prefix))

    # Try progressively shorter suffixes
    for trim in range(10, min(60, len(child_stripped)), 10):
        suffix = child_stripped[:-trim]
        if len(suffix) < 40:
            break
        idx = parent_text.find(suffix)
        if idx != -1:
            return (idx, idx + len(suffix))

    return None


def _compute_overlap(
    window_start: int,
    window_end: int,
    child_start: int,
    child_end: int,
) -> float:
    """Compute fraction of the keyword window that overlaps with the child span."""
    window_len = window_end - window_start
    if window_len <= 0:
        return 1.0  # degenerate: treat as overlapping to skip

    overlap_start = max(window_start, child_start)
    overlap_end = min(window_end, child_end)
    overlap_len = max(0, overlap_end - overlap_start)

    return overlap_len / window_len


def _expand_to_sentence_boundaries(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Expand a character range to the nearest sentence boundaries."""
    # Expand start backward to sentence boundary
    expanded_start = start
    search_back = text[:start]
    for pattern in ['. ', '.\n', '!\n', '! ', '?\n', '? ']:
        idx = search_back.rfind(pattern)
        if idx != -1:
            expanded_start = min(expanded_start, idx + len(pattern))
            break
    else:
        # No sentence boundary found — expand to paragraph or start
        para_idx = search_back.rfind('\n\n')
        if para_idx != -1:
            expanded_start = para_idx + 2
        else:
            expanded_start = 0

    # Expand end forward to sentence boundary
    expanded_end = end
    search_forward = text[end:]
    for pattern in ['. ', '.\n', '!\n', '! ', '?\n', '? ']:
        idx = search_forward.find(pattern)
        if idx != -1:
            candidate = end + idx + 1  # include the period/punctuation
            expanded_end = max(expanded_end, candidate)
            break
    else:
        # No sentence boundary — expand to paragraph or end
        para_idx = search_forward.find('\n\n')
        if para_idx != -1:
            expanded_end = end + para_idx
        else:
            expanded_end = len(text)

    return (expanded_start, expanded_end)


def compute_highlight_texts(
    answer_text: str,
    retrieval_results: list,
) -> dict[int, str]:
    """Compute corrected highlight texts for citations that need them.

    Parameters
    ----------
    answer_text : str
        The full generated answer text (with citation markers).
    retrieval_results : list
        List of ``RetrievalResult`` objects from the retrieval pipeline.

    Returns
    -------
    dict[int, str]
        Mapping of ``{citation_number: highlight_text}`` only for citations
        whose referenced content falls outside the child chunk boundaries.
        Empty dict means all citations are fine as-is.
    """
    if not answer_text or not retrieval_results:
        return {}

    claims = _extract_claims(answer_text)
    if not claims:
        return {}

    highlight_map: dict[int, str] = {}

    for cit_num, claim_sentences in claims.items():
        # Citation numbers in the answer are 1-indexed; retrieval results are 0-indexed
        result_idx = cit_num - 1
        if result_idx < 0 or result_idx >= len(retrieval_results):
            continue

        result = retrieval_results[result_idx]

        # Skip if no parent text (no context expansion)
        if not result.parent_text:
            continue

        child_text = result.text or ""
        parent_text = result.parent_text

        # Collect stems from all claim sentences for this citation
        all_stems: list[str] = []
        for sentence in claim_sentences:
            all_stems.extend(_extract_stems(sentence))

        if not all_stems:
            continue

        # Find where stems cluster in the parent text
        hits = _find_stem_positions(all_stems, parent_text)
        if not hits:
            continue

        window = _best_density_window(hits)
        if window is None:
            continue

        window_start, window_end, _ = window

        # Check overlap with child chunk
        child_span = _find_child_span(child_text, parent_text)
        if child_span is not None:
            overlap = _compute_overlap(
                window_start, window_end,
                child_span[0], child_span[1],
            )
            # If >70% overlaps with child chunk, current highlighting is fine
            if overlap > 0.70:
                continue

        # Extract highlight text expanded to sentence boundaries
        expanded_start, expanded_end = _expand_to_sentence_boundaries(
            parent_text, window_start, window_end,
        )
        highlight_text = parent_text[expanded_start:expanded_end].strip()

        # Only use if we got something meaningful
        if len(highlight_text) >= 20:
            highlight_map[cit_num] = highlight_text

    return highlight_map
