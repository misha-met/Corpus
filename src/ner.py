"""
Offline NER for place name extraction at ingest time.
Uses GLiNER medium v2.1 directly via the gliner package.
Falls back to regex (extract_places_from_query) if GLiNER is unavailable.

GLiNER achieves F1 ~0.85-0.89 on general NER vs ~0.67 for regex heuristics,
meaningfully reducing false positives on person names and titles.
"""
from __future__ import annotations

import logging
import re
import threading
import warnings
from dataclasses import dataclass
from typing import TypedDict

log = logging.getLogger(__name__)

_GLINER_MODEL = "urchade/gliner_medium-v2.1"
# City-only extraction to avoid broad location spans such as countries/regions.
_NER_LABELS = ["city"]
_NER_THRESHOLD = 0.4
_BATCH_SIZE = 16
_TOKEN_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['-])+", flags=re.UNICODE)
_NONSPACE_RE = re.compile(r"\S+")
_GLINER_MAX_SEQUENCE_TOKENS = 384
_GLINER_SAFE_SEQUENCE_TOKENS = 300
_GLINER_WINDOW_TOKENS = 250
_GLINER_WINDOW_OVERLAP_TOKENS = 50
_TOKEN_ESTIMATE_MULTIPLIER = 1.6
_NER_METHOD_GLINER = "gliner"
_NER_METHOD_REGEX_FALLBACK = "regex_fallback"
_NER_METHOD_EMPTY = "empty"
_PLACE_LABELS = {"CITY", "LOCATION", "LOC", "GPE"}
_PERSON_LABELS = {"PERSON", "PER"}
_PERSON_BLOCKLIST = {
    "figure",
    "table",
    "chapter",
    "section",
    "appendix",
    "references",
    "reference",
    "bibliography",
    "introduction",
    "conclusion",
    "abstract",
    "supplement",
}
_PERSON_TOKEN_RE = re.compile(r"^[^\W\d_][^\W\d_\-'.]*$", flags=re.UNICODE)

_model = None
_model_lock = threading.Lock()
_model_ready = False


class PlaceCandidate(TypedDict):
    text: str
    entity_type: str
    score: float
    start: int
    end: int
    context_words: list[str]


class PersonCandidate(TypedDict):
    text: str
    entity_type: str
    score: float
    start: int
    end: int
    context_words: list[str]


class _EntityCandidate(TypedDict):
    text: str
    entity_type: str
    score: float
    start: int
    end: int


@dataclass(frozen=True)
class NERDiagnostics:
    """Runtime diagnostics describing which NER path was used.

    ``ner_available`` reports whether GLiNER was available for inference,
    ``method`` records the extraction strategy used by the caller, and
    ``warning`` provides a human-readable degradation reason when applicable.
    """

    ner_available: bool
    method: str
    warning: str | None = None


def _get_model():
    global _model, _model_ready
    if _model_ready:
        return _model
    with _model_lock:
        if _model_ready:
            return _model
        try:
            from gliner import GLiNER

            _model = GLiNER.from_pretrained(_GLINER_MODEL)
            _model_ready = True
            log.info("GLiNER model loaded (%s).", _GLINER_MODEL)
        except Exception as exc:
            log.warning("GLiNER unavailable — NER will fall back to regex: %s", exc)
            _model = None
            _model_ready = True
    return _model


def _extract_context_words(text: str, start: int, end: int, window_words: int) -> list[str]:
    if window_words <= 0:
        return []
    left_text = text[max(0, start - 240):max(0, start)]
    right_text = text[min(len(text), end):min(len(text), end + 240)]

    left_tokens = _TOKEN_RE.findall(left_text)
    right_tokens = _TOKEN_RE.findall(right_text)

    left_take = max(1, window_words // 2)
    right_take = max(0, window_words - left_take)
    return left_tokens[-left_take:] + right_tokens[:right_take]


def _coerce_bounds(text: str, candidate: str, raw_start: object, raw_end: object) -> tuple[int, int]:
    if isinstance(raw_start, (int, float)):
        ent_start = int(raw_start)
    else:
        # Fallback: search all occurrences and warn if ambiguous instead of silently
        # using the first occurrence, which can be wrong for repeated mentions.
        positions = [m.start() for m in re.finditer(re.escape(candidate), text)]
        if not positions:
            ent_start = 0
        elif len(positions) == 1:
            ent_start = positions[0]
        else:
            log.warning(
                "Ambiguous bounds for candidate %r (%d occurrences); using first occurrence.",
                candidate,
                len(positions),
            )
            ent_start = positions[0]

    ent_end = int(raw_end) if isinstance(raw_end, (int, float)) else ent_start + len(candidate)
    ent_end = max(ent_start, min(len(text), ent_end))
    return ent_start, ent_end


def _estimate_token_count(text: str, *, model: object | None = None) -> int:
    """Estimate GLiNER token count using tokenizer when accessible, else rough heuristic."""
    if not text:
        return 0

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                encoded = encode(text, add_special_tokens=False)
                if isinstance(encoded, list):
                    return len(encoded)
            except TypeError:
                try:
                    encoded = encode(text)
                    if isinstance(encoded, list):
                        return len(encoded)
                except Exception:
                    pass
            except Exception:
                pass

        if callable(tokenizer):
            try:
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                if isinstance(encoded, dict):
                    input_ids = encoded.get("input_ids")
                    if isinstance(input_ids, list):
                        if input_ids and isinstance(input_ids[0], list):
                            return len(input_ids[0])
                        return len(input_ids)
            except Exception:
                pass

    words = len(text.split())
    word_estimate = int(words * _TOKEN_ESTIMATE_MULTIPLIER)
    char_estimate = int(len(text) / 3.2)
    return max(1, word_estimate, char_estimate)


def _is_gliner_truncation_warning(message: object) -> bool:
    text = str(message).lower()
    return "truncated to" in text and str(_GLINER_MAX_SEQUENCE_TOKENS) in text


def _predict_entities_with_warning_capture(
    model: object,
    text_or_texts: str | list[str],
    labels: list[str],
    threshold: float,
) -> tuple[object, bool]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        entities = model.predict_entities(text_or_texts, labels, threshold=threshold)
    saw_truncation = any(_is_gliner_truncation_warning(warning.message) for warning in captured)
    return entities, saw_truncation


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _NONSPACE_RE.finditer(text)]


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _dedupe_overlap_entities(rows: list[_EntityCandidate]) -> list[_EntityCandidate]:
    """Deduplicate overlap-window duplicates by text + overlapping span, keeping highest score."""
    deduped: list[_EntityCandidate] = []
    for row in sorted(rows, key=lambda item: (item["start"], item["end"])):
        lowered = row["text"].lower()
        overlapping_idx: int | None = None

        for idx, existing in enumerate(deduped):
            if existing["text"].lower() != lowered:
                continue
            if _spans_overlap(existing["start"], existing["end"], row["start"], row["end"]):
                overlapping_idx = idx
                break

        if overlapping_idx is None:
            deduped.append(row)
            continue

        existing = deduped[overlapping_idx]
        if row["score"] > existing["score"]:
            deduped[overlapping_idx] = row
        elif row["score"] == existing["score"]:
            existing_span = existing["end"] - existing["start"]
            row_span = row["end"] - row["start"]
            if row_span > existing_span:
                deduped[overlapping_idx] = row

    return sorted(deduped, key=lambda item: (item["start"], item["end"]))


def _predict_entities_windowed(
    model: object,
    text: str,
    *,
    labels: list[str],
    threshold: float,
) -> list[_EntityCandidate]:
    """Predict entities with safe sliding-window inference to avoid GLiNER max-length truncation."""
    if not text.strip():
        return []

    word_count = len(text.split())
    estimated_tokens = _estimate_token_count(text, model=model)
    if estimated_tokens <= _GLINER_SAFE_SEQUENCE_TOKENS and word_count <= _GLINER_SAFE_SEQUENCE_TOKENS:
        entities, saw_truncation = _predict_entities_with_warning_capture(
            model,
            text,
            labels,
            threshold,
        )
        if saw_truncation:
            entities = []
        else:
            rows: list[_EntityCandidate] = []
            for ent in entities:
                candidate = str(ent.get("text", "")).strip()
                if not candidate:
                    continue
                ent_start, ent_end = _coerce_bounds(text, candidate, ent.get("start"), ent.get("end"))
                score_raw = ent.get("score", 0.0)
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0
                rows.append(
                    {
                        "text": candidate,
                        "entity_type": str(ent.get("label", "")).upper().strip() or "UNKNOWN",
                        "score": score,
                        "start": ent_start,
                        "end": ent_end,
                    }
                )
            return rows

    spans = _token_spans(text)
    if not spans:
        return []

    total_words = len(spans)
    if _GLINER_WINDOW_OVERLAP_TOKENS >= _GLINER_WINDOW_TOKENS:
        raise ValueError("Window overlap must be smaller than window size.")

    start_idx = 0
    rows: list[_EntityCandidate] = []

    while start_idx < total_words:
        end_idx = min(start_idx + _GLINER_WINDOW_TOKENS, total_words)
        window_start_char = spans[start_idx][0]

        # Use a binary search to find the largest end index within this window
        # whose estimated token count does not exceed the safe maximum, instead
        # of shrinking the window one token at a time.
        lo = start_idx + 1
        hi = end_idx
        best_end_idx = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_end_char = spans[mid - 1][1]
            window_text = text[window_start_char:mid_end_char]
            if _estimate_token_count(window_text, model=model) <= _GLINER_SAFE_SEQUENCE_TOKENS:
                best_end_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        current_end_idx = best_end_idx
        while True:
            window_end_char = spans[current_end_idx - 1][1]
            window_text = text[window_start_char:window_end_char]
            entities, saw_truncation = _predict_entities_with_warning_capture(
                model,
                window_text,
                labels,
                threshold,
            )
            if not saw_truncation or current_end_idx <= start_idx + 1:
                end_idx = current_end_idx
                break
            span_size = current_end_idx - start_idx
            reduced_span = max(1, span_size // 2)
            current_end_idx = start_idx + reduced_span

        for ent in entities:
            candidate = str(ent.get("text", "")).strip()
            if not candidate:
                continue

            local_start, local_end = _coerce_bounds(
                window_text,
                candidate,
                ent.get("start"),
                ent.get("end"),
            )
            score_raw = ent.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0

            rows.append(
                {
                    "text": candidate,
                    "entity_type": str(ent.get("label", "")).upper().strip() or "UNKNOWN",
                    "score": score,
                    "start": window_start_char + local_start,
                    "end": window_start_char + local_end,
                }
            )

        if end_idx >= total_words:
            break

        next_start = max(start_idx + 1, end_idx - _GLINER_WINDOW_OVERLAP_TOKENS)
        start_idx = next_start

    return _dedupe_overlap_entities(rows)


def _predict_entity_candidates(
    texts: list[str],
    *,
    labels: list[str],
    threshold: float,
) -> list[list[_EntityCandidate]]:
    def _rows_for_text(text: str, entities_for_text: list[dict]) -> list[_EntityCandidate]:
        rows: list[_EntityCandidate] = []
        for ent in entities_for_text or []:
            candidate = str(ent.get("text", "")).strip()
            if not candidate:
                continue
            ent_start, ent_end = _coerce_bounds(
                text,
                candidate,
                ent.get("start"),
                ent.get("end"),
            )
            score_raw = ent.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            rows.append(
                {
                    "text": candidate,
                    "entity_type": str(ent.get("label", "")).upper().strip() or "UNKNOWN",
                    "score": score,
                    "start": ent_start,
                    "end": ent_end,
                }
            )
        return rows

    model = _get_model()
    if model is None:
        raise RuntimeError("GLiNER model unavailable")

    n = len(texts)
    if n == 0:
        return []

    # Pre-compute which texts are short enough for single-pass inference without
    # windowing so we can use true batched GLiNER inference where safe.
    word_counts = [len(text.split()) for text in texts]
    token_estimates = [_estimate_token_count(text, model=model) for text in texts]
    needs_window = [
        token_estimates[i] > _GLINER_SAFE_SEQUENCE_TOKENS or word_counts[i] > _GLINER_SAFE_SEQUENCE_TOKENS
        for i in range(n)
    ]

    results: list[list[_EntityCandidate]] = [[] for _ in range(n)]
    indices = list(range(n))

    for batch_start in range(0, n, _BATCH_SIZE):
        batch_indices = indices[batch_start : batch_start + _BATCH_SIZE]

        # Short texts: try true batched inference.
        short_indices = [i for i in batch_indices if not needs_window[i] and texts[i].strip()]
        if short_indices:
            batch_texts = [texts[i] for i in short_indices]
            try:
                raw_batch_entities, saw_batch_truncation = _predict_entities_with_warning_capture(
                    model,
                    batch_texts,
                    labels,
                    threshold,
                )
            except TypeError:
                # Older GLiNER versions may not support list inputs; fall back to per-text.
                for i in short_indices:
                    results[i] = _predict_entities_windowed(
                        model,
                        texts[i],
                        labels=labels,
                        threshold=threshold,
                    )
            else:
                batch_entities: list[list[dict]] | None = None
                # GLiNER should return a list of lists (one per input text). If we
                # instead get a flat list or a non-list type, treat that as an
                # unsupported batch mode and fall back to per-text windowed inference
                # for this batch to avoid mis-aligning entities to texts.
                if not isinstance(raw_batch_entities, list):
                    batch_entities = None
                elif raw_batch_entities and all(
                    not isinstance(item, list) for item in raw_batch_entities
                ):
                    batch_entities = None
                else:
                    batch_entities = [
                        ents if isinstance(ents, list) else [] for ents in raw_batch_entities
                    ]

                if batch_entities is None:
                    for i in short_indices:
                        results[i] = _predict_entities_windowed(
                            model,
                            texts[i],
                            labels=labels,
                            threshold=threshold,
                        )
                elif saw_batch_truncation:
                    # Detect truncation at per-text granularity and only rerun
                    # the affected items through windowed inference.
                    for idx_in_batch, i in enumerate(short_indices):
                        single_entities, saw_text_truncation = _predict_entities_with_warning_capture(
                            model,
                            texts[i],
                            labels,
                            threshold,
                        )
                        if saw_text_truncation:
                            results[i] = _predict_entities_windowed(
                                model,
                                texts[i],
                                labels=labels,
                                threshold=threshold,
                            )
                            continue

                        if isinstance(single_entities, list):
                            entities_for_text = single_entities
                        else:
                            entities_for_text = (
                                batch_entities[idx_in_batch] if idx_in_batch < len(batch_entities) else []
                            )
                        results[i] = _rows_for_text(texts[i], entities_for_text)
                else:
                    for idx_in_batch, i in enumerate(short_indices):
                        entities_for_text = (
                            batch_entities[idx_in_batch] if idx_in_batch < len(batch_entities) else []
                        )
                        results[i] = _rows_for_text(texts[i], entities_for_text)

        # Long texts: fall back to windowed inference, which already dedupes overlapping
        # mentions but preserves distinct mentions at different offsets.
        long_indices = [i for i in batch_indices if needs_window[i] and texts[i].strip()]
        for i in long_indices:
            results[i] = _predict_entities_windowed(
                model,
                texts[i],
                labels=labels,
                threshold=threshold,
            )

    return results


def _to_place_candidates(
    text: str,
    rows: list[_EntityCandidate],
    *,
    min_score: float,
    context_window_words: int,
) -> list[PlaceCandidate]:
    candidates: list[PlaceCandidate] = []
    for row in rows:
        entity_type = row["entity_type"]
        if entity_type not in _PLACE_LABELS:
            continue
        if row["score"] < min_score:
            continue
        candidates.append(
            {
                "text": row["text"],
                "entity_type": entity_type,
                "score": row["score"],
                "start": row["start"],
                "end": row["end"],
                "context_words": _extract_context_words(
                    text,
                    row["start"],
                    row["end"],
                    context_window_words,
                ),
            }
        )
    return candidates


def _is_predominantly_lowercase(text: str, threshold: float = 0.1) -> bool:
    """Return True if fewer than `threshold` fraction of alphabetic characters are uppercase."""
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return False
    upper_count = sum(1 for ch in alpha_chars if ch.isupper())
    return (upper_count / len(alpha_chars)) < threshold


def _looks_like_person_name(value: str, *, source_text: str | None = None) -> bool:
    text = value.strip()
    if len(text) < 3:
        return False
    if any(char.isdigit() for char in text):
        return False
    if text.lower() in _PERSON_BLOCKLIST:
        return False

    # If the surrounding source text appears predominantly lowercase (e.g., OCR
    # or chat logs), relax the casing requirement; otherwise, continue to reject
    # all-lowercase candidate strings to reduce false positives like section
    # headings or generic nouns.
    if source_text is None or not _is_predominantly_lowercase(source_text):
        if text == text.lower():
            return False

    tokens = [token for token in re.split(r"\s+", text) if token]
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False

    for token in tokens:
        if token.lower() in _PERSON_BLOCKLIST:
            return False
        # Require that person-name tokens start with a letter and contain at
        # least one alphabetic character overall, but allow Unicode letters and
        # common punctuation such as hyphens and apostrophes.
        if not any(ch.isalpha() for ch in token):
            return False
        if not token[0].isalpha():
            return False
        if any(ch.isdigit() for ch in token):
            return False
        if not _PERSON_TOKEN_RE.match(token):
            return False
    return True


def _to_person_candidates(
    text: str,
    rows: list[_EntityCandidate],
    *,
    min_score: float,
    context_window_words: int,
) -> list[PersonCandidate]:
    candidates: list[PersonCandidate] = []
    for row in rows:
        entity_type = row["entity_type"]
        if entity_type not in _PERSON_LABELS:
            continue
        if row["score"] < min_score:
            continue
        if not _looks_like_person_name(row["text"], source_text=text):
            continue
        candidates.append(
            {
                "text": row["text"],
                "entity_type": entity_type,
                "score": row["score"],
                "start": row["start"],
                "end": row["end"],
                "context_words": _extract_context_words(
                    text,
                    row["start"],
                    row["end"],
                    context_window_words,
                ),
            }
        )
    return candidates


def _build_regex_place_fallback(
    texts: list[str],
    *,
    context_window_words: int,
) -> list[list[PlaceCandidate]]:
    from .geocoder import extract_places_from_query

    fallback: list[list[PlaceCandidate]] = []
    for text in texts:
        rows: list[PlaceCandidate] = []
        for candidate in extract_places_from_query(text):
            start = text.find(candidate)
            if start < 0:
                start = 0
            end = min(len(text), start + len(candidate))
            rows.append(
                {
                    "text": candidate,
                    "entity_type": "LOCATION",
                    "score": 0.0,
                    "start": start,
                    "end": end,
                    "context_words": _extract_context_words(text, start, end, context_window_words),
                }
            )
        fallback.append(rows)
    return fallback


def extract_place_candidates_ner_with_diagnostics(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> tuple[list[list[PlaceCandidate]], NERDiagnostics]:
    """Extract place candidates and report whether GLiNER was used."""
    try:
        predicted = _predict_entity_candidates(
            texts,
            labels=_NER_LABELS,
            threshold=threshold,
        )
        return (
            [
                _to_place_candidates(
                    text,
                    rows,
                    min_score=threshold,
                    context_window_words=context_window_words,
                )
                for text, rows in zip(texts, predicted)
            ],
            NERDiagnostics(ner_available=True, method=_NER_METHOD_GLINER),
        )
    except Exception as exc:
        warning = f"GLiNER place inference unavailable; fell back to regex ({type(exc).__name__}: {exc})"
        log.warning(warning)
        return (
            _build_regex_place_fallback(texts, context_window_words=context_window_words),
            NERDiagnostics(
                ner_available=False,
                method=_NER_METHOD_REGEX_FALLBACK,
                warning=warning,
            ),
        )


def extract_place_candidates_ner(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> list[list[PlaceCandidate]]:
    """Extract place candidates with metadata for downstream geocoding decisions."""
    rows, _diagnostics = extract_place_candidates_ner_with_diagnostics(
        texts,
        threshold=threshold,
        context_window_words=context_window_words,
    )
    return rows


def extract_person_candidates_ner_with_diagnostics(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> tuple[list[list[PersonCandidate]], NERDiagnostics]:
    """Extract person candidates and report whether GLiNER was used."""
    try:
        predicted = _predict_entity_candidates(
            texts,
            labels=["person"],
            threshold=threshold,
        )
        return (
            [
                _to_person_candidates(
                    text,
                    rows,
                    min_score=threshold,
                    context_window_words=context_window_words,
                )
                for text, rows in zip(texts, predicted)
            ],
            NERDiagnostics(ner_available=True, method=_NER_METHOD_GLINER),
        )
    except Exception as exc:
        warning = (
            "GLiNER person inference unavailable; returning empty person candidates "
            f"({type(exc).__name__}: {exc})"
        )
        log.warning(warning)
        return (
            [[] for _ in texts],
            NERDiagnostics(
                ner_available=False,
                method=_NER_METHOD_EMPTY,
                warning=warning,
            ),
        )


def extract_person_candidates_ner(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> list[list[PersonCandidate]]:
    """Extract person candidates for resolver canonicalization."""
    rows, _diagnostics = extract_person_candidates_ner_with_diagnostics(
        texts,
        threshold=threshold,
        context_window_words=context_window_words,
    )
    return rows


def extract_place_and_person_candidates_ner_with_diagnostics(
    texts: list[str],
    *,
    geo_threshold: float,
    people_threshold: float,
    geo_context_window_words: int = 8,
    people_context_window_words: int = 8,
) -> tuple[list[list[PlaceCandidate]], list[list[PersonCandidate]], NERDiagnostics, NERDiagnostics]:
    """Run joint geo/person extraction and return diagnostics for each output."""
    shared_threshold = min(float(geo_threshold), float(people_threshold))
    try:
        predicted = _predict_entity_candidates(
            texts,
            labels=["city", "person"],
            threshold=shared_threshold,
        )
        place_rows = [
            _to_place_candidates(
                text,
                rows,
                min_score=float(geo_threshold),
                context_window_words=geo_context_window_words,
            )
            for text, rows in zip(texts, predicted)
        ]
        person_rows = [
            _to_person_candidates(
                text,
                rows,
                min_score=float(people_threshold),
                context_window_words=people_context_window_words,
            )
            for text, rows in zip(texts, predicted)
        ]
        gliner_diag = NERDiagnostics(ner_available=True, method=_NER_METHOD_GLINER)
        return place_rows, person_rows, gliner_diag, gliner_diag
    except Exception as exc:
        warning = (
            "GLiNER shared geo/person inference failed; using regex fallback for places "
            f"and empty people output ({type(exc).__name__}: {exc})"
        )
        log.warning(warning)
        return (
            _build_regex_place_fallback(texts, context_window_words=geo_context_window_words),
            [[] for _ in texts],
            NERDiagnostics(
                ner_available=False,
                method=_NER_METHOD_REGEX_FALLBACK,
                warning=warning,
            ),
            NERDiagnostics(
                ner_available=False,
                method=_NER_METHOD_EMPTY,
                warning=warning,
            ),
        )


def extract_place_and_person_candidates_ner(
    texts: list[str],
    *,
    geo_threshold: float,
    people_threshold: float,
    geo_context_window_words: int = 8,
    people_context_window_words: int = 8,
) -> tuple[list[list[PlaceCandidate]], list[list[PersonCandidate]]]:
    """Run a single GLiNER pass for city+person labels and post-filter by per-type thresholds."""
    place_rows, person_rows, _place_diag, _person_diag = (
        extract_place_and_person_candidates_ner_with_diagnostics(
            texts,
            geo_threshold=geo_threshold,
            people_threshold=people_threshold,
            geo_context_window_words=geo_context_window_words,
            people_context_window_words=people_context_window_words,
        )
    )
    return place_rows, person_rows


def extract_places_ner(texts: list[str]) -> list[list[str]]:
    """Backward-compatible helper returning only candidate text strings."""
    return [[candidate["text"] for candidate in row] for row in extract_place_candidates_ner(texts)]
