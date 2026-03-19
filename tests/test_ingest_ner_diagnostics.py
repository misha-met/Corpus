"""Regression tests for ingest-time NER diagnostics propagation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest import ingest_file_to_storage
from src.storage import StorageConfig, StorageEngine


class _DummyEmbeddingModel:
    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        _ = normalize_embeddings
        return [[0.01] * 16 for _ in texts]


def test_place_ner_fallback_reports_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ner import extract_place_candidates_ner_with_diagnostics

    def _raise_predict(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("GLiNER unavailable")

    monkeypatch.setattr("src.ner._predict_entity_candidates", _raise_predict)

    rows, diagnostics = extract_place_candidates_ner_with_diagnostics(["Paris and Rome"])

    assert diagnostics.ner_available is False
    assert diagnostics.method == "regex_fallback"
    assert len(rows) == 1


def test_person_ner_unavailable_reports_empty_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ner import extract_person_candidates_ner_with_diagnostics

    def _raise_predict(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("GLiNER unavailable")

    monkeypatch.setattr("src.ner._predict_entity_candidates", _raise_predict)

    rows, diagnostics = extract_person_candidates_ner_with_diagnostics(["Noam Chomsky wrote this."])

    assert diagnostics.ner_available is False
    assert diagnostics.method == "empty"
    assert rows == [[]]


def test_ingest_file_to_storage_returns_degraded_ner_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "doc.md"
    source_file.write_text(
        "# Notes\n\nNoam Chomsky wrote about Paris and Rome in this document.",
        encoding="utf-8",
    )

    def _raise_predict(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("GLiNER unavailable")

    monkeypatch.setattr("src.ner._predict_entity_candidates", _raise_predict)

    storage = StorageEngine(StorageConfig(lance_dir=tmp_path / "lance"))
    try:
        parents_count, children_count, diagnostics = ingest_file_to_storage(
            source_file,
            source_id="diag_doc",
            page_number=None,
            storage=storage,
            embedding_model=_DummyEmbeddingModel(),
            summarize=False,
            geotag=True,
            peopletag=True,
            page_offset=1,
        )
    finally:
        storage.close()

    assert parents_count >= 1
    assert children_count >= 1
    assert diagnostics.geotag_ner is not None
    assert diagnostics.peopletag_ner is not None
    assert diagnostics.geotag_ner.ner_available is False
    assert diagnostics.geotag_ner.method == "regex_fallback"
    assert diagnostics.peopletag_ner.ner_available is False
    assert diagnostics.peopletag_ner.method == "empty"
