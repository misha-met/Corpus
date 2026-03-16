from __future__ import annotations

import pytest

from src.ner import _predict_entities_windowed


def _build_text(total_tokens: int, replacements: dict[int, str]) -> str:
    tokens = [f"w{i}" for i in range(total_tokens)]
    for idx, value in replacements.items():
        tokens[idx] = value
    return " ".join(tokens)


class _TruncatingModel:
    """Simulates GLiNER behavior where only the early part of long text is processed."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, object]]:
        _ = labels, threshold
        self.calls.append(text)

        # Simulated truncation: model only "sees" the first ~320 words.
        visible = " ".join(text.split()[:320])
        rows: list[dict[str, object]] = []
        for marker, score in (("Paris", 0.9), ("Rome", 0.82)):
            start = visible.find(marker)
            if start < 0:
                continue
            rows.append(
                {
                    "text": marker,
                    "label": "city",
                    "score": score,
                    "start": start,
                    "end": start + len(marker),
                }
            )
        return rows


class _OverlapDuplicateModel:
    """Returns the same entity from adjacent windows with different confidence."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, object]]:
        _ = labels, threshold
        self.calls += 1
        marker = "Alexandria"
        start = text.find(marker)
        if start < 0:
            return []

        # Deterministic overlap duplicate: first seen is lower confidence,
        # second seen is higher confidence and should win deduplication.
        score = 0.56 if self.calls == 1 else 0.93
        return [
            {
                "text": marker,
                "label": "city",
                "score": score,
                "start": start,
                "end": start + len(marker),
            }
        ]


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, object]]:
        _ = labels, threshold
        self.calls += 1
        marker = "Paris"
        start = text.find(marker)
        if start < 0:
            return []
        return [
            {
                "text": marker,
                "label": "city",
                "score": 0.8,
                "start": start,
                "end": start + len(marker),
            }
        ]


def test_windowed_prediction_captures_entities_near_tail() -> None:
    model = _TruncatingModel()
    text = _build_text(420, {20: "Paris", 390: "Rome"})

    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    names = {row["text"] for row in rows}
    assert "Paris" in names
    assert "Rome" in names
    assert len(model.calls) >= 2

    for row in rows:
        assert text[row["start"]:row["end"]] == row["text"]


def test_overlap_dedup_keeps_higher_confidence_entity() -> None:
    model = _OverlapDuplicateModel()
    text = _build_text(360, {280: "Alexandria"})

    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    alexandria_rows = [row for row in rows if row["text"] == "Alexandria"]
    assert len(alexandria_rows) == 1
    assert alexandria_rows[0]["score"] == pytest.approx(0.93)
    assert model.calls >= 2


def test_short_text_uses_single_predict_call() -> None:
    model = _CountingModel()
    text = _build_text(100, {40: "Paris"})

    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    assert len(rows) == 1
    assert model.calls == 1
