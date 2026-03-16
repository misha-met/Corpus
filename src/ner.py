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
from typing import TypedDict

log = logging.getLogger(__name__)

_GLINER_MODEL = "urchade/gliner_medium-v2.1"
# City-only extraction to avoid broad location spans such as countries/regions.
_NER_LABELS = ["city"]
_NER_THRESHOLD = 0.4
_BATCH_SIZE = 16
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")
_NONSPACE_RE = re.compile(r"\S+")
_GLINER_MAX_SEQUENCE_TOKENS = 384
_GLINER_SAFE_SEQUENCE_TOKENS = 350
_GLINER_WINDOW_TOKENS = 300
_GLINER_WINDOW_OVERLAP_TOKENS = 50
_TOKEN_ESTIMATE_MULTIPLIER = 1.3
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
_PERSON_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z\-'.]*$")

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
    ent_start = int(raw_start) if isinstance(raw_start, (int, float)) else text.find(candidate)
    if ent_start < 0:
        ent_start = 0

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
    return max(1, int(words * _TOKEN_ESTIMATE_MULTIPLIER))


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
        entities = model.predict_entities(text, labels, threshold=threshold)
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
        window_end_char = spans[end_idx - 1][1]
        window_text = text[window_start_char:window_end_char]

        # Tighten window if token estimation still exceeds safe maximum.
        while end_idx - start_idx > 1 and _estimate_token_count(window_text, model=model) > _GLINER_SAFE_SEQUENCE_TOKENS:
            end_idx -= 1
            window_end_char = spans[end_idx - 1][1]
            window_text = text[window_start_char:window_end_char]

        entities = model.predict_entities(window_text, labels, threshold=threshold)
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
    model = _get_model()
    if model is None:
        raise RuntimeError("GLiNER model unavailable")

    results: list[list[_EntityCandidate]] = []
    for start_idx in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start_idx : start_idx + _BATCH_SIZE]
        for text in batch:
            entities = _predict_entities_windowed(
                model,
                text,
                labels=labels,
                threshold=threshold,
            )
            # Preserve first-seen order while keeping the highest score per (label, text).
            ordered_keys: list[tuple[str, str]] = []
            by_key: dict[tuple[str, str], _EntityCandidate] = {}
            for ent in entities:
                candidate = str(ent.get("text", "")).strip()
                if not candidate:
                    continue
                label = str(ent.get("entity_type", "")).upper().strip() or "UNKNOWN"

                row: _EntityCandidate = {
                    "text": candidate,
                    "entity_type": label,
                    "score": float(ent.get("score", 0.0)),
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                }

                key = (label, candidate.lower())
                if key not in by_key:
                    ordered_keys.append(key)
                    by_key[key] = row
                    continue

                if row["score"] > by_key[key]["score"]:
                    by_key[key] = row

            results.append([by_key[key] for key in ordered_keys])
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


def _looks_like_person_name(value: str) -> bool:
    text = value.strip()
    if len(text) < 3:
        return False
    if any(char.isdigit() for char in text):
        return False
    if text.lower() in _PERSON_BLOCKLIST:
        return False
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
        if not _looks_like_person_name(row["text"]):
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


def extract_place_candidates_ner(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> list[list[PlaceCandidate]]:
    """Extract place candidates with metadata for downstream geocoding decisions."""
    try:
        predicted = _predict_entity_candidates(
            texts,
            labels=_NER_LABELS,
            threshold=threshold,
        )
        return [
            _to_place_candidates(
                text,
                rows,
                min_score=threshold,
                context_window_words=context_window_words,
            )
            for text, rows in zip(texts, predicted)
        ]
    except Exception as exc:
        log.warning("GLiNER inference failed, falling back to regex: %s", exc)

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


def extract_person_candidates_ner(
    texts: list[str],
    *,
    threshold: float = _NER_THRESHOLD,
    context_window_words: int = 8,
) -> list[list[PersonCandidate]]:
    """Extract person candidates for resolver canonicalization."""
    try:
        predicted = _predict_entity_candidates(
            texts,
            labels=["person"],
            threshold=threshold,
        )
        return [
            _to_person_candidates(
                text,
                rows,
                min_score=threshold,
                context_window_words=context_window_words,
            )
            for text, rows in zip(texts, predicted)
        ]
    except Exception as exc:
        log.warning("GLiNER person inference failed; returning no person candidates: %s", exc)
        return [[] for _ in texts]


def extract_place_and_person_candidates_ner(
    texts: list[str],
    *,
    geo_threshold: float,
    people_threshold: float,
    geo_context_window_words: int = 8,
    people_context_window_words: int = 8,
) -> tuple[list[list[PlaceCandidate]], list[list[PersonCandidate]]]:
    """Run a single GLiNER pass for city+person labels and post-filter by per-type thresholds."""
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
        return place_rows, person_rows
    except Exception as exc:
        log.warning("GLiNER shared geo/person inference failed; falling back for places only: %s", exc)
        return extract_place_candidates_ner(texts, context_window_words=geo_context_window_words), [[] for _ in texts]


def extract_places_ner(texts: list[str]) -> list[list[str]]:
    """Backward-compatible helper returning only candidate text strings."""
    return [[candidate["text"] for candidate in row] for row in extract_place_candidates_ner(texts)]
