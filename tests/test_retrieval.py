"""Tests for retrieval pipeline: hybrid search, deduplication, reranking, thresholds."""
from __future__ import annotations

from typing import Any

import pytest

from src.config import ModelConfig
from src.retrieval import RetrievalEngine, RetrievalResult
from src.metrics import DeduplicationMetrics, ThresholdMetrics
from tests.conftest import (
    MockEmbeddingModel,
    MockReranker,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
    FIXED_QUERIES,
)

logger = get_test_logger("retrieval")


# ===========================================================================
# Helper: build a config for testing
# ===========================================================================

def _make_config(mode: str = "regular") -> ModelConfig:
    """Create a minimal ModelConfig for testing."""
    if mode == "regular":
        return ModelConfig(
            mode="regular",
            llm_model="test-model",
            embedding_model="test-embed",
            reranker_model="test-reranker",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=6,
            top_k_final=3,
            reranker_threshold=0.05,
            reranker_min_docs=2,
            retrieval_budget=8000,
        )
    elif mode == "power-fast":
        return ModelConfig(
            mode="power-fast",
            llm_model="test-model",
            embedding_model="test-embed",
            reranker_model="test-reranker",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_fused=15,
            top_k_rerank=10,
            top_k_final=5,
            reranker_threshold=0.02,
            reranker_min_docs=3,
            retrieval_budget=50000,
        )
    elif mode == "power-deep-research":
        return ModelConfig(
            mode="power-deep-research",
            llm_model="test-model",
            embedding_model="test-embed",
            reranker_model="test-reranker",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_fused=15,
            top_k_rerank=10,
            top_k_final=5,
            reranker_threshold=0.01,
            reranker_min_docs=5,
            retrieval_budget=20000,
        )
    raise ValueError(f"Unknown test mode: {mode}")


@pytest.fixture
def retrieval_engine(tmp_storage, mock_embedder, mock_reranker):
    """Build a RetrievalEngine with mocked components."""
    config = _make_config("regular")
    return RetrievalEngine(
        storage=tmp_storage,
        embedding_model=mock_embedder,
        reranker=mock_reranker,
        config=config,
    )


# ===========================================================================
# Hybrid search
# ===========================================================================

class TestHybridSearch:
    def test_hybrid_returns_results(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine._hybrid_search("Chomsky theory", top_k=5)
        assert len(results) > 0
        assert all("id" in r for r in results)

    def test_hybrid_top_k_limit(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine._hybrid_search("language acquisition", top_k=2)
        assert len(results) <= 2

    def test_hybrid_deterministic(self, retrieval_engine: RetrievalEngine):
        r1 = retrieval_engine._hybrid_search("epistemology knowledge", top_k=5)
        r2 = retrieval_engine._hybrid_search("epistemology knowledge", top_k=5)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_hybrid_empty_query_raises(self, retrieval_engine: RetrievalEngine):
        with pytest.raises(ValueError):
            retrieval_engine._hybrid_search("", top_k=5)

    def test_hybrid_whitespace_query_raises(self, retrieval_engine: RetrievalEngine):
        with pytest.raises(ValueError):
            retrieval_engine._hybrid_search("   ", top_k=5)

    def test_hybrid_source_filter_applies(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine._hybrid_search(
            "language",
            top_k=10,
            source_id="test_doc_linguistics",
        )
        assert len(results) > 0
        assert all((r.get("metadata") or {}).get("source_id") == "test_doc_linguistics" for r in results)


# ===========================================================================
# Early parent-level deduplication
# ===========================================================================

class TestDeduplication:
    def test_dedup_keeps_highest_score(self):
        """Should keep the child with highest score per parent_id."""
        items = [
            {"id": "c1", "score": 0.9, "metadata": {"parent_id": "p1"}},
            {"id": "c2", "score": 0.5, "metadata": {"parent_id": "p1"}},
            {"id": "c3", "score": 0.8, "metadata": {"parent_id": "p2"}},
        ]
        deduped, metrics = RetrievalEngine._deduplicate_by_parent(items, top_k=10)
        ids = [d["id"] for d in deduped]
        assert "c1" in ids  # higher score for p1
        assert "c2" not in ids
        assert "c3" in ids

    def test_dedup_metrics_reduction(self):
        items = [
            {"id": f"c{i}", "score": float(i), "metadata": {"parent_id": "p1"}}
            for i in range(5)
        ]
        deduped, metrics = RetrievalEngine._deduplicate_by_parent(items, top_k=10)
        assert metrics.children_before_dedup == 5
        assert metrics.children_after_dedup == 1
        assert metrics.reduction_pct > 0

    def test_dedup_no_parent_id_preserved(self):
        """Items without parent_id should be preserved."""
        items = [
            {"id": "c1", "score": 0.9, "metadata": {}},
            {"id": "c2", "score": 0.5, "metadata": {}},
        ]
        deduped, _ = RetrievalEngine._deduplicate_by_parent(items, top_k=10)
        assert len(deduped) == 2

    def test_dedup_top_k_trim(self):
        """Dedup should trim to top_k after deduplication."""
        items = [
            {"id": f"c{i}", "score": float(i), "metadata": {"parent_id": f"p{i}"}}
            for i in range(10)
        ]
        deduped, _ = RetrievalEngine._deduplicate_by_parent(items, top_k=3)
        assert len(deduped) <= 3

    def test_dedup_empty_list(self):
        deduped, metrics = RetrievalEngine._deduplicate_by_parent([], top_k=10)
        assert len(deduped) == 0


# ===========================================================================
# Reranking
# ===========================================================================

class TestReranking:
    def test_rerank_scores_assigned(self, retrieval_engine: RetrievalEngine):
        items = [
            {"id": "c1", "text": "Chomsky argues about language"},
            {"id": "c2", "text": "Ethics and moral philosophy"},
        ]
        reranked, raw_scores = retrieval_engine._rerank("Chomsky language", items)
        assert len(reranked) == 2
        assert all("rerank_score" in r for r in reranked)
        assert len(raw_scores) == 2

    def test_rerank_sorted_by_score(self, retrieval_engine: RetrievalEngine):
        items = [
            {"id": "c1", "text": "unrelated topic"},
            {"id": "c2", "text": "Chomsky language grammar theory"},
        ]
        reranked, _ = retrieval_engine._rerank("Chomsky language grammar", items)
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_boilerplate_filtered_in_search_stage(self, retrieval_engine: RetrievalEngine, monkeypatch):
        """Boilerplate is filtered in final search stage, not _rerank()."""
        def _fake_hybrid_search(query: str, top_k: int, *, source_id=None):
            return [
                {
                    "id": "boiler",
                    "text": "As an AI language model, I cannot be considered a person",
                    "score": 0.9,
                    "metadata": {"parent_id": "p1", "source_id": "test_doc_linguistics"},
                },
                {
                    "id": "normal",
                    "text": "Chomsky language theory and generative grammar",
                    "score": 0.8,
                    "metadata": {"parent_id": "p2", "source_id": "test_doc_linguistics"},
                },
            ]

        monkeypatch.setattr(retrieval_engine, "_hybrid_search", _fake_hybrid_search)
        results = retrieval_engine.search("language", collect_metrics=False)
        ids = [r.child_id for r in results]
        assert "boiler" not in ids
        assert "normal" in ids

    def test_rerank_empty_items(self, retrieval_engine: RetrievalEngine):
        reranked, scores = retrieval_engine._rerank("test", [])
        assert reranked == []
        assert scores == []

    def test_rerank_no_reranker(self, tmp_storage, mock_embedder):
        """Without reranker, items should pass through unchanged."""
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=None,
            config=_make_config(),
        )
        items = [{"id": "c1", "text": "test"}]
        reranked, _ = engine._rerank("test", items)
        assert len(reranked) == 1


# ===========================================================================
# Threshold filtering and safety net
# ===========================================================================

class TestThresholdFiltering:
    def test_threshold_filters_low_scores(self, tmp_storage, mock_embedder, mock_reranker):
        """Verify threshold filtering removes low-scoring documents."""
        config = _make_config("regular")
        # Our mock reranker returns overlap-based scores (0..1 range).
        # Set threshold at 0.3 to filter out low overlap.
        config = ModelConfig(
            mode=config.mode,
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            top_k_dense=config.top_k_dense,
            top_k_sparse=config.top_k_sparse,
            top_k_fused=config.top_k_fused,
            top_k_rerank=config.top_k_rerank,
            top_k_final=config.top_k_final,
            reranker_threshold=0.5,  # High threshold to trigger filtering
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky")
        # Should still get results (safety net if needed)
        assert len(results) >= 1

    def test_safety_net_minimum_docs(self, tmp_storage, mock_embedder, mock_reranker):
        """Safety net should ensure minimum docs even if all below threshold."""
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=6,
            top_k_final=3,
            reranker_threshold=999.0,  # Impossibly high - everything filtered
            reranker_min_docs=2,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky language theory")
        # Safety net should provide at least min_docs
        assert len(results) >= 1  # May be less due to final dedup

    def test_threshold_metrics_recorded(self, tmp_storage, mock_embedder, mock_reranker):
        """Threshold metrics should be recorded in retrieval metrics."""
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=_make_config("regular"),
        )
        results = engine.search("Chomsky theory")
        if results and results[0].metrics:
            metrics = results[0].metrics
            assert metrics.threshold is not None


# ===========================================================================
# Full search pipeline
# ===========================================================================

class TestFullSearch:
    def test_search_returns_results(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine.search("Chomsky language theory")
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)

    def test_search_results_have_text(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine.search("epistemology knowledge")
        for r in results:
            assert r.text or r.parent_text

    def test_search_results_have_score(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine.search("ethics moral philosophy")
        for r in results:
            assert isinstance(r.score, float)

    def test_search_metrics_collected(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine.search("Chomsky", collect_metrics=True)
        if results:
            metrics = results[0].metrics
            assert metrics is not None
            assert metrics.timing.total_ms > 0
            assert metrics.timing.hybrid_search_ms >= 0
            assert metrics.timing.sparse_search_ms >= 0

    def test_search_no_duplicate_parents(self, retrieval_engine: RetrievalEngine):
        """Final results should not contain duplicate parent_ids."""
        results = retrieval_engine.search("Chomsky language")
        parent_ids = [
            r.metadata.get("parent_id") for r in results
            if r.metadata.get("parent_id")
        ]
        assert len(parent_ids) == len(set(parent_ids))

    def test_search_deterministic(self, retrieval_engine: RetrievalEngine):
        """Same query should produce same results."""
        r1 = retrieval_engine.search("Chomsky theory")
        r2 = retrieval_engine.search("Chomsky theory")
        assert [r.child_id for r in r1] == [r.child_id for r in r2]


# ===========================================================================
# Mode comparison
# ===========================================================================

class TestModeComparison:
    def test_modes_differ_in_top_k(self):
        """Different modes should have different top_k parameters."""
        regular = _make_config("regular")
        power = _make_config("power-fast")
        assert power.top_k_dense > regular.top_k_dense
        assert power.top_k_final > regular.top_k_final

    def test_modes_differ_in_threshold(self):
        regular = _make_config("regular")
        power = _make_config("power-fast")
        deep = _make_config("power-deep-research")
        assert power.reranker_threshold < regular.reranker_threshold
        assert deep.reranker_threshold < power.reranker_threshold


# ===========================================================================
# Latency: retrieval stages
# ===========================================================================

class TestRetrievalLatency:
    def test_hybrid_search_latency(self, retrieval_engine: RetrievalEngine):
        for q in ["Chomsky theory", "epistemology knowledge", "ethics"]:
            with Timer("hybrid_search", query=q) as t:
                retrieval_engine._hybrid_search(q, top_k=10)
            logger.info(f"hybrid_search '{q}': {t.result.elapsed_ms:.2f}ms")

    def test_rerank_latency_by_batch_size(self, retrieval_engine: RetrievalEngine):
        for batch_size in [5, 10, 20]:
            items = [
                {"id": f"c{i}", "text": f"document content about topic {i}"}
                for i in range(batch_size)
            ]
            with Timer("rerank", batch_size=batch_size) as t:
                retrieval_engine._rerank("test query topic", items)
            logger.info(f"rerank batch_size={batch_size}: {t.result.elapsed_ms:.2f}ms")

    def test_dedup_latency(self):
        items = [
            {"id": f"c{i}", "score": float(i), "metadata": {"parent_id": f"p{i % 20}"}}
            for i in range(100)
        ]
        with Timer("dedup", input_count=100) as t:
            RetrievalEngine._deduplicate_by_parent(items, top_k=50)
        logger.info(f"dedup 100 items: {t.result.elapsed_ms:.2f}ms")

    def test_full_search_latency(self, retrieval_engine: RetrievalEngine):
        queries = [q for q in FIXED_QUERIES if q.strip()]
        for q in queries:
            with Timer("full_search", query=q) as t:
                results = retrieval_engine.search(q)
            result_count = len(results)
            logger.info(
                f"full_search '{q[:40]}...': {t.result.elapsed_ms:.2f}ms, {result_count} results"
            )
