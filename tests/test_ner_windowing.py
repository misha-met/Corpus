from __future__ import annotations

import warnings

import pytest

from src.ner import _predict_entities_windowed, _predict_entity_candidates


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


class _AdaptiveWarningModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, object]]:
        _ = labels, threshold
        word_count = len(text.split())
        self.calls.append(word_count)
        if word_count > 120:
            warnings.warn("Sentence of length 507 has been truncated to 384", UserWarning)
            visible = " ".join(text.split()[:120])
        else:
            visible = text
        rows: list[dict[str, object]] = []
        for marker in ("Paris", "Rome"):
            start = visible.find(marker)
            if start < 0:
                continue
            rows.append(
                {
                    "text": marker,
                    "label": "city",
                    "score": 0.9,
                    "start": start,
                    "end": start + len(marker),
                }
            )
        return rows


class _BatchWarningModel:
    def predict_entities(
        self,
        text_or_texts: str | list[str],
        labels: list[str],
        threshold: float,
    ) -> list[dict[str, object]] | list[list[dict[str, object]]]:
        _ = labels, threshold
        if isinstance(text_or_texts, list):
            warnings.warn("Sentence of length 625 has been truncated to 384", UserWarning)
            return [[] for _ in text_or_texts]
        marker = "Paris"
        start = text_or_texts.find(marker)
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


class _BoundaryStraddleModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, object]]:
        _ = labels, threshold
        self.calls += 1
        marker = "New York"
        start = text.find(marker)
        if start < 0:
            return []
        return [
            {
                "text": marker,
                "label": "city",
                "score": 0.9,
                "start": start,
                "end": start + len(marker),
            }
        ]


class _SelectiveBatchWarningModel:
    def predict_entities(
        self,
        text_or_texts: str | list[str],
        labels: list[str],
        threshold: float,
    ) -> list[dict[str, object]] | list[list[dict[str, object]]]:
        _ = labels, threshold
        if isinstance(text_or_texts, list):
            warnings.warn("Sentence of length 625 has been truncated to 384", UserWarning)
            return [
                [
                    {
                        "text": "Paris",
                        "label": "city",
                        "score": 0.91,
                        "start": text_or_texts[0].find("Paris"),
                        "end": text_or_texts[0].find("Paris") + len("Paris"),
                    }
                ],
                [
                    {
                        "text": "Rome",
                        "label": "city",
                        "score": 0.67,
                        "start": text_or_texts[1].find("Rome"),
                        "end": text_or_texts[1].find("Rome") + len("Rome"),
                    }
                ],
            ]

        text = text_or_texts
        if "TRUNCATE" in text:
            warnings.warn("Sentence of length 625 has been truncated to 384", UserWarning)

        marker = "Rome" if "Rome" in text else "Paris"
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


def test_window_overlap_captures_boundary_straddle_entity() -> None:
    model = _BoundaryStraddleModel()
    text = _build_text(420, {249: "New", 250: "York"})

    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    matching = [row for row in rows if row["text"] == "New York"]
    assert len(matching) == 1
    row = matching[0]
    assert text[row["start"]:row["end"]] == "New York"
    assert model.calls >= 2


def test_short_text_uses_single_predict_call() -> None:
    model = _CountingModel()
    text = _build_text(100, {40: "Paris"})

    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    assert len(rows) == 1
    assert model.calls == 1


def test_windowed_prediction_retries_smaller_windows_on_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _AdaptiveWarningModel()
    text = _build_text(170, {20: "Paris", 150: "Rome"})

    monkeypatch.setattr("src.ner._estimate_token_count", lambda _text, model=None: 100)
    rows = _predict_entities_windowed(model, text, labels=["city"], threshold=0.1)

    names = {row["text"] for row in rows}
    assert names == {"Paris", "Rome"}
    assert max(model.calls) > 120
    assert min(model.calls) <= 120


def test_batch_truncation_warning_falls_back_to_per_text_windowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _BatchWarningModel()
    monkeypatch.setattr("src.ner._get_model", lambda: model)

    predicted = _predict_entity_candidates(["alpha Paris beta"], labels=["city"], threshold=0.1)

    assert len(predicted) == 1
    assert [row["text"] for row in predicted[0]] == ["Paris"]


def test_batch_truncation_reruns_only_truncated_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SelectiveBatchWarningModel()
    monkeypatch.setattr("src.ner._get_model", lambda: model)

    windowed_calls: list[str] = []

    def _fake_windowed(_model, text: str, *, labels: list[str], threshold: float):
        _ = labels, threshold
        windowed_calls.append(text)
        start = text.find("Rome")
        return [
            {
                "text": "Rome",
                "entity_type": "CITY",
                "score": 0.99,
                "start": start,
                "end": start + len("Rome"),
            }
        ]

    monkeypatch.setattr("src.ner._predict_entities_windowed", _fake_windowed)

    texts = [
        "alpha Paris beta",
        "alpha TRUNCATE Rome beta",
    ]
    predicted = _predict_entity_candidates(texts, labels=["city"], threshold=0.1)

    assert windowed_calls == [texts[1]]
    assert [row["text"] for row in predicted[0]] == ["Paris"]
    assert [row["text"] for row in predicted[1]] == ["Rome"]
    assert predicted[1][0]["score"] == pytest.approx(0.99)
