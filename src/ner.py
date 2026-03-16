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


def extract_place_candidates_ner(
    texts: list[str],
    *,
    context_window_words: int = 8,
) -> list[list[PlaceCandidate]]:
    """Extract place candidates with metadata for downstream geocoding decisions."""
    model = _get_model()
    if model is not None:
        try:
            results: list[list[PlaceCandidate]] = []
            for start_idx in range(0, len(texts), _BATCH_SIZE):
                batch = texts[start_idx : start_idx + _BATCH_SIZE]
                for text in batch:
                    entities = model.predict_entities(
                        text,
                        _NER_LABELS,
                        threshold=_NER_THRESHOLD,
                    )
                    # Preserve first-seen order while keeping the highest score per exact text.
                    ordered_keys: list[str] = []
                    by_text: dict[str, PlaceCandidate] = {}
                    for ent in entities:
                        candidate = str(ent.get("text", "")).strip()
                        if not candidate:
                            continue
                        lowered = candidate.lower()

                        raw_start = ent.get("start")
                        raw_end = ent.get("end")
                        ent_start = int(raw_start) if isinstance(raw_start, (int, float)) else text.find(candidate)
                        if ent_start < 0:
                            ent_start = 0
                        ent_end = int(raw_end) if isinstance(raw_end, (int, float)) else ent_start + len(candidate)
                        ent_end = max(ent_start, min(len(text), ent_end))

                        score_raw = ent.get("score", 0.0)
                        try:
                            score = float(score_raw)
                        except (TypeError, ValueError):
                            score = 0.0

                        entity_type = str(ent.get("label", "LOCATION") or "LOCATION").upper()
                        row: PlaceCandidate = {
                            "text": candidate,
                            "entity_type": entity_type,
                            "score": score,
                            "start": ent_start,
                            "end": ent_end,
                            "context_words": _extract_context_words(text, ent_start, ent_end, context_window_words),
                        }

                        if lowered not in by_text:
                            ordered_keys.append(lowered)
                            by_text[lowered] = row
                            continue

                        # Keep whichever duplicate has the higher NER score.
                        if row["score"] > by_text[lowered]["score"]:
                            by_text[lowered] = row

                    results.append([by_text[key] for key in ordered_keys])
            return results
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


def extract_places_ner(texts: list[str]) -> list[list[str]]:
    """Backward-compatible helper returning only candidate text strings."""
    return [[candidate["text"] for candidate in row] for row in extract_place_candidates_ner(texts)]
