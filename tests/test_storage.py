"""Tests for unified LanceDB storage engine."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.models import ChildChunk, Metadata, ParentChunk
from src.storage import StorageConfig, StorageEngine
from tests.conftest import (
    MockEmbeddingModel,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
)

logger = get_test_logger("storage")


# ===========================================================================
# Parent store (LanceDB)
# ===========================================================================

class TestParentStore:
    def test_add_and_retrieve_parent(self, tmp_storage: StorageEngine):
        """Stored parent text should be retrievable by parent_id."""
        sources = tmp_storage.list_source_ids()
        assert len(sources) > 0, "Should have source_ids after ingest"
        texts = tmp_storage.get_parent_texts_by_source(source_id=sources[0])
        assert len(texts) > 0, "Should have parent texts for first source"

    def test_missing_parent_returns_none(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_parent_text("nonexistent-id") is None

    def test_list_source_ids(self, tmp_storage: StorageEngine):
        sources = tmp_storage.list_source_ids()
        assert "test_doc_linguistics" in sources
        assert "test_doc_philosophy" in sources

    def test_get_parent_texts_by_source(self, tmp_storage: StorageEngine):
        texts = tmp_storage.get_parent_texts_by_source(source_id="test_doc_linguistics")
        assert len(texts) > 0


# ===========================================================================
# Child chunk store (LanceDB vectors + FTS)
# ===========================================================================

class TestChildStore:
    def test_hybrid_search_returns_results(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Hybrid search should return child chunks."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=5,
        )
        assert len(results) > 0

    def test_hybrid_search_top_k_respected(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Returned results should not exceed top_k."""
        q_vec = mock_embedder.encode(["test query"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="test query", query_vector=q_vec, top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_search_returns_metadata(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Each result should include metadata with source_id."""
        q_vec = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="Chomsky grammar", query_vector=q_vec, top_k=3,
        )
        for r in results:
            assert "source_id" in r["metadata"]

    def test_get_children_by_ids(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """get_children_by_ids should return text and metadata."""
        q_vec = mock_embedder.encode(["test"], normalize_embeddings=True)[0]
        search_results = tmp_storage.hybrid_search(
            query_text="test", query_vector=q_vec, top_k=3,
        )
        ids = [r["id"] for r in search_results]
        fetched = tmp_storage.get_children_by_ids(ids)
        assert len(fetched) == len(ids)
        for cid, data in fetched.items():
            assert "text" in data
            assert "metadata" in data

    def test_empty_ids_returns_empty(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_children_by_ids([]) == {}

    def test_hybrid_search_deterministic(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Same query should produce same results on repeated calls."""
        q_vec = mock_embedder.encode(["Chomsky language"], normalize_embeddings=True)[0]
        r1 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        r2 = tmp_storage.hybrid_search(
            query_text="Chomsky language", query_vector=q_vec, top_k=5,
        )
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_hybrid_search_source_filter(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Source filter should restrict results to one source."""
        q_vec = mock_embedder.encode(["knowledge epistemology"], normalize_embeddings=True)[0]
        results = tmp_storage.hybrid_search(
            query_text="knowledge epistemology", query_vector=q_vec,
            top_k=10, source_id="test_doc_philosophy",
        )
        for r in results:
            assert r["metadata"].get("source_id") == "test_doc_philosophy"

    def test_persist_reopen(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        """Data should survive close + reopen."""
        config = StorageConfig(lance_dir=tmp_path / "lance")
        engine = StorageEngine(config)
        parents, children = generate_parent_child_corpus()
        engine.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = mock_embedder.encode(texts, normalize_embeddings=True)
        engine.add_children(children, embeddings=embeddings)
        engine.close()

        engine2 = StorageEngine(config)
        sources = engine2.list_source_ids()
        assert len(sources) > 0, "Data should persist across sessions"
        engine2.close()


# ===========================================================================
# Source summaries (LanceDB)
# ===========================================================================

class TestSourceSummaries:
    def test_upsert_and_get_summary(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="test_doc", summary="A test summary.")
        summaries = tmp_storage.get_source_summaries()
        assert "test_doc" in summaries
        assert summaries["test_doc"] == "A test summary."

    def test_upsert_overwrites(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v1")
        tmp_storage.upsert_source_summary(source_id="doc_x", summary="v2")
        summaries = tmp_storage.get_source_summaries()
        assert summaries["doc_x"] == "v2"

    def test_empty_source_id_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="", summary="text")

    def test_empty_summary_rejected(self, tmp_storage: StorageEngine):
        with pytest.raises(ValueError):
            tmp_storage.upsert_source_summary(source_id="doc", summary="")


# ===========================================================================
# Latency: index operations
# ===========================================================================

class TestStorageLatency:
    def test_hybrid_search_latency(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Measure hybrid search latency."""
        q_vec = mock_embedder.encode(["Chomsky theory"], normalize_embeddings=True)[0]
        top_k_values = [5, 10, 20]
        for k in top_k_values:
            with Timer("hybrid_search", top_k=k) as t:
                tmp_storage.hybrid_search(
                    query_text="Chomsky theory", query_vector=q_vec, top_k=k,
                )
            logger.info(f"hybrid_search top_k={k}: {t.result.elapsed_ms:.2f}ms")

    def test_parent_lookup_latency(self, tmp_storage: StorageEngine):
        """Measure parent text lookup latency."""
        parents, _ = generate_parent_child_corpus()
        for parent in parents[:3]:
            with Timer("parent_lookup", parent_id=parent.id) as t:
                tmp_storage.get_parent_text(parent.id)
            logger.info(f"parent_lookup: {t.result.elapsed_ms:.2f}ms")
