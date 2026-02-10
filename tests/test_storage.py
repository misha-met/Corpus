"""Tests for storage engine: vector index, BM25, SQLite context store."""
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
# SQLite parent store
# ===========================================================================

class TestParentStore:
    def test_add_and_retrieve_parent(self, tmp_storage: StorageEngine):
        """Stored parent text should be retrievable by parent_id."""
        # Query to get actual child IDs stored in this fixture
        embedder = MockEmbeddingModel()
        q_emb = embedder.encode(["test"], normalize_embeddings=True)
        results = tmp_storage.query_children(embeddings=q_emb, top_k=10)
        metadatas = results.get("metadatas", [[]])[0]
        parent_ids = {m.get("parent_id") for m in metadatas if m.get("parent_id")}
        assert len(parent_ids) > 0, "Should have parent_ids in metadata"
        for pid in parent_ids:
            text = tmp_storage.get_parent_text(pid)
            assert text is not None, f"Parent {pid} should be retrievable"
            assert len(text) > 0

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
# Vector (Chroma) store
# ===========================================================================

class TestVectorStore:
    def test_query_returns_results(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Vector query should return child chunks."""
        q_embedding = mock_embedder.encode(["test query"], normalize_embeddings=True)
        results = tmp_storage.query_children(embeddings=q_embedding, top_k=5)
        ids = results.get("ids", [[]])[0]
        assert len(ids) > 0

    def test_query_top_k_respected(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Returned results should not exceed top_k."""
        q_embedding = mock_embedder.encode(["test query"], normalize_embeddings=True)
        results = tmp_storage.query_children(embeddings=q_embedding, top_k=2)
        ids = results.get("ids", [[]])[0]
        assert len(ids) <= 2

    def test_query_returns_metadata(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Each result should include metadata with source_id."""
        q_embedding = mock_embedder.encode(["Chomsky grammar"], normalize_embeddings=True)
        results = tmp_storage.query_children(embeddings=q_embedding, top_k=3)
        metadatas = results.get("metadatas", [[]])[0]
        for meta in metadatas:
            assert "source_id" in meta

    def test_get_children_by_ids(self, tmp_storage: StorageEngine):
        """get_children_by_ids should return text and metadata."""
        # First get some IDs from a query
        parents, children = generate_parent_child_corpus()
        embedder = MockEmbeddingModel()
        q_emb = embedder.encode(["test"], normalize_embeddings=True)
        results = tmp_storage.query_children(embeddings=q_emb, top_k=3)
        ids = results.get("ids", [[]])[0]
        
        fetched = tmp_storage.get_children_by_ids(ids)
        assert len(fetched) == len(ids)
        for cid, data in fetched.items():
            assert "text" in data
            assert "metadata" in data

    def test_empty_ids_returns_empty(self, tmp_storage: StorageEngine):
        assert tmp_storage.get_children_by_ids([]) == {}

    def test_deterministic_retrieval(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Same query should produce same results on repeated calls."""
        q_emb = mock_embedder.encode(["Chomsky language"], normalize_embeddings=True)
        r1 = tmp_storage.query_children(embeddings=q_emb, top_k=5)
        r2 = tmp_storage.query_children(embeddings=q_emb, top_k=5)
        assert r1["ids"] == r2["ids"]
        assert r1["distances"] == r2["distances"]


# ===========================================================================
# BM25 sparse index
# ===========================================================================

class TestBM25Index:
    def test_bm25_initialized(self, tmp_storage: StorageEngine):
        assert tmp_storage.bm25 is not None

    def test_bm25_ids_populated(self, tmp_storage: StorageEngine):
        assert len(tmp_storage.bm25_ids) > 0

    def test_bm25_search_returns_scores(self, tmp_storage: StorageEngine):
        bm25 = tmp_storage.bm25
        assert bm25 is not None
        scores = bm25.get_scores("Chomsky language grammar".split())
        assert len(scores) > 0
        assert max(scores) > 0

    def test_bm25_persist_reload(self, tmp_path: Path, mock_embedder: MockEmbeddingModel):
        """BM25 index should survive persist/reload cycle."""
        config = StorageConfig(
            sqlite_path=tmp_path / "test.sqlite",
            chroma_dir=tmp_path / "chroma",
        )
        engine = StorageEngine(config)
        parents, children = generate_parent_child_corpus()
        engine.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = mock_embedder.encode(texts, normalize_embeddings=True)
        engine.add_children(children, embeddings=embeddings)

        bm25_path = tmp_path / "bm25.json"
        engine.persist_bm25(bm25_path)

        engine2 = StorageEngine(config)
        engine2.load_bm25(bm25_path)

        assert engine2.bm25 is not None
        assert len(engine2.bm25_ids) == len(engine.bm25_ids)

        engine.close()
        engine2.close()

    def test_bm25_source_ids_aligned(self, tmp_storage: StorageEngine):
        """BM25 source_ids should be aligned with bm25_ids."""
        assert len(tmp_storage.bm25_ids) == len(tmp_storage.bm25_source_ids)

    def test_bm25_load_missing_file_raises(self, tmp_storage: StorageEngine, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            tmp_storage.load_bm25(tmp_path / "nonexistent.json")


# ===========================================================================
# Source summaries
# ===========================================================================

class TestSourceSummaries:
    def test_upsert_and_get_summary(self, tmp_storage: StorageEngine):
        tmp_storage.upsert_source_summary(source_id="test_doc", summary="A test summary.")
        summaries = tmp_storage.get_source_summaries()
        assert "test_doc" in summaries
        assert summaries["test_doc"] == "A test summary."

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
    def test_vector_query_latency(self, tmp_storage: StorageEngine, mock_embedder: MockEmbeddingModel):
        """Measure vector query latency."""
        q_emb = mock_embedder.encode(["Chomsky theory"], normalize_embeddings=True)
        top_k_values = [5, 10, 20]
        for k in top_k_values:
            with Timer("vector_query", top_k=k) as t:
                tmp_storage.query_children(embeddings=q_emb, top_k=k)
            logger.info(f"vector_query top_k={k}: {t.result.elapsed_ms:.2f}ms")

    def test_bm25_search_latency(self, tmp_storage: StorageEngine):
        """Measure BM25 search latency."""
        queries = ["Chomsky language", "epistemology knowledge", "ethics moral"]
        for q in queries:
            bm25 = tmp_storage.bm25
            with Timer("bm25_search", query=q) as t:
                bm25.get_scores(q.split())
            logger.info(f"bm25_search query='{q}': {t.result.elapsed_ms:.2f}ms")

    def test_parent_lookup_latency(self, tmp_storage: StorageEngine):
        """Measure SQLite parent text lookup latency."""
        parents, _ = generate_parent_child_corpus()
        for parent in parents[:3]:
            with Timer("parent_lookup", parent_id=parent.id) as t:
                tmp_storage.get_parent_text(parent.id)
            logger.info(f"parent_lookup: {t.result.elapsed_ms:.2f}ms")
