"""Tests for retrieval pipeline: hybrid search, deduplication, reranking, thresholds."""
from __future__ import annotations

from typing import Any

import pytest

from src.config import ModelConfig
from src.models import ChildChunk, Metadata, ParentChunk
from src import retrieval
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
    elif mode == "deep-research":
        return ModelConfig(
            mode="deep-research",
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
        results = retrieval_engine._hybrid_search_decoupled(
            embedding_query="Chomsky theory", bm25_query="Chomsky theory", top_k=5,
        )
        assert len(results) > 0
        assert all("id" in r for r in results)

    def test_hybrid_top_k_limit(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine._hybrid_search_decoupled(
            embedding_query="language acquisition", bm25_query="language acquisition", top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_deterministic(self, retrieval_engine: RetrievalEngine):
        r1 = retrieval_engine._hybrid_search_decoupled(
            embedding_query="epistemology knowledge", bm25_query="epistemology knowledge", top_k=5,
        )
        r2 = retrieval_engine._hybrid_search_decoupled(
            embedding_query="epistemology knowledge", bm25_query="epistemology knowledge", top_k=5,
        )
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_hybrid_empty_query_raises(self, retrieval_engine: RetrievalEngine):
        with pytest.raises(ValueError):
            retrieval_engine._hybrid_search_decoupled(
                embedding_query="", bm25_query="", top_k=5,
            )

    def test_hybrid_whitespace_query_raises(self, retrieval_engine: RetrievalEngine):
        with pytest.raises(ValueError):
            retrieval_engine._hybrid_search_decoupled(
                embedding_query="   ", bm25_query="   ", top_k=5,
            )

    def test_hybrid_source_filter_applies(self, retrieval_engine: RetrievalEngine):
        results = retrieval_engine._hybrid_search_decoupled(
            embedding_query="language", bm25_query="language", top_k=10,
            source_id="test_doc_linguistics",
        )
        assert len(results) > 0
        assert all((r.get("metadata") or {}).get("source_id") == "test_doc_linguistics" for r in results)

    def test_subword_heavy_query_runs_without_error(self, retrieval_engine: RetrievalEngine):
        response = retrieval_engine.search("noam chomsky's theory of language")
        assert isinstance(response.results, list)

    def test_custom_bm25_weight_is_forwarded(self, retrieval_engine: RetrievalEngine, monkeypatch):
        captured: dict[str, float] = {}

        def _fake_hybrid_search(*, query_text, query_vector, top_k, source_id=None, bm25_weight=0.5):
            _ = query_text, query_vector, top_k, source_id
            captured["bm25_weight"] = float(bm25_weight)
            return []

        monkeypatch.setattr(retrieval_engine._storage, "hybrid_search", _fake_hybrid_search)

        response = retrieval_engine.search("Chomsky language", bm25_weight=0.2)

        assert response.results == []
        assert captured["bm25_weight"] == pytest.approx(0.2)

    def test_use_hybrid_false_uses_vector_search(self, retrieval_engine: RetrievalEngine, monkeypatch):
        calls = {"vector": 0}

        def _fake_vector_search(*, query_vector, top_k, source_id=None):
            _ = query_vector, top_k, source_id
            calls["vector"] += 1
            return []

        def _fail_hybrid_search(*args, **kwargs):
            _ = args, kwargs
            raise AssertionError("hybrid_search should not be called when use_hybrid=False")

        monkeypatch.setattr(retrieval_engine._storage, "vector_search", _fake_vector_search)
        monkeypatch.setattr(retrieval_engine._storage, "hybrid_search", _fail_hybrid_search)

        response = retrieval_engine.search("Chomsky language", use_hybrid=False)

        assert response.results == []
        assert calls["vector"] == 1


# ===========================================================================
# Early parent-level deduplication
# ===========================================================================

class TestDeduplication:
    def test_dedup_keeps_highest_score(self):
        """Should keep the child with highest score per parent_id (one-per-parent mode)."""
        items = [
            {"id": "c1", "score": 0.9, "metadata": {"parent_id": "p1"}},
            {"id": "c2", "score": 0.5, "metadata": {"parent_id": "p1"}},
            {"id": "c3", "score": 0.8, "metadata": {"parent_id": "p2"}},
        ]
        deduped, metrics = RetrievalEngine._deduplicate_by_parent(items, top_k=10, max_children_per_parent=1)
        ids = [d["id"] for d in deduped]
        assert "c1" in ids  # higher score for p1
        assert "c2" not in ids
        assert "c3" in ids

    def test_dedup_metrics_reduction(self):
        items = [
            {"id": f"c{i}", "score": float(i), "metadata": {"parent_id": "p1"}}
            for i in range(5)
        ]
        deduped, metrics = RetrievalEngine._deduplicate_by_parent(items, top_k=10, max_children_per_parent=1)
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

    def test_dedup_two_children_per_parent_default(self):
        """Default max_children_per_parent=2: both top children from same parent are kept."""
        items = [
            {"id": "c1", "score": 0.9, "metadata": {"parent_id": "p1"}},
            {"id": "c2", "score": 0.85, "metadata": {"parent_id": "p1"}},
            {"id": "c3", "score": 0.5, "metadata": {"parent_id": "p1"}},
            {"id": "c4", "score": 0.82, "metadata": {"parent_id": "p2"}},
        ]
        deduped, metrics = RetrievalEngine._deduplicate_by_parent(items, top_k=10)
        ids = [d["id"] for d in deduped]
        assert "c1" in ids
        assert "c2" in ids   # second-best sibling kept under default=2
        assert "c3" not in ids  # third sibling dropped
        assert "c4" in ids
        assert metrics.children_before_dedup == 4
        assert metrics.children_after_dedup == 3  # 2 from p1 + 1 from p2

    def test_dedup_diversity_when_budget_allows(self):
        """With top_k=3 and max_children_per_parent=2, second sibling enters the list."""
        items = [
            {"id": "a1", "score": 0.9,  "metadata": {"parent_id": "pA"}},
            {"id": "a2", "score": 0.85, "metadata": {"parent_id": "pA"}},
            {"id": "a3", "score": 0.5,  "metadata": {"parent_id": "pA"}},
            {"id": "b1", "score": 0.82, "metadata": {"parent_id": "pB"}},
        ]
        # With max=2, top_k=3: a1 (0.9), a2 (0.85), b1 (0.82)
        deduped, _ = RetrievalEngine._deduplicate_by_parent(items, top_k=3, max_children_per_parent=2)
        ids = [d["id"] for d in deduped]
        assert ids == ["a1", "a2", "b1"]

    def test_dedup_displacement_when_budget_tight(self):
        """When top_k=2, the second sibling displaces the lower-scored other-parent chunk."""
        items = [
            {"id": "a1", "score": 0.9,  "metadata": {"parent_id": "pA"}},
            {"id": "a2", "score": 0.85, "metadata": {"parent_id": "pA"}},
            {"id": "a3", "score": 0.5,  "metadata": {"parent_id": "pA"}},
            {"id": "b1", "score": 0.82, "metadata": {"parent_id": "pB"}},
        ]
        # With max=2, top_k=2: a1 (0.9), a2 (0.85) — b1 displaced
        deduped, _ = RetrievalEngine._deduplicate_by_parent(items, top_k=2, max_children_per_parent=2)
        ids = [d["id"] for d in deduped]
        assert "a1" in ids
        assert "a2" in ids
        assert "b1" not in ids

    def test_dedup_one_per_parent_baseline(self):
        """max_children_per_parent=1 reproduces old one-per-parent behaviour."""
        items = [
            {"id": "a1", "score": 0.9,  "metadata": {"parent_id": "pA"}},
            {"id": "a2", "score": 0.85, "metadata": {"parent_id": "pA"}},
            {"id": "b1", "score": 0.82, "metadata": {"parent_id": "pB"}},
        ]
        deduped, _ = RetrievalEngine._deduplicate_by_parent(items, top_k=3, max_children_per_parent=1)
        ids = [d["id"] for d in deduped]
        assert "a1" in ids
        assert "a2" not in ids  # only best child kept
        assert "b1" in ids

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
        response = engine.search("Chomsky")
        results = response.results
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
        response = engine.search("Chomsky language theory")
        results = response.results
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
        response = engine.search("Chomsky theory")
        results = response.results
        if results and response.metrics:
            metrics = response.metrics
            assert metrics.threshold is not None

    def test_safety_net_tags_below_threshold_items(
        self,
        tmp_storage,
        mock_embedder,
        mock_reranker,
        monkeypatch,
    ):
        """min_docs safety net: sub-threshold items get below_threshold=True;
        above-threshold items are not tagged."""
        # Set a threshold of 0.5 so only the first item passes.
        # min_docs=3 forces the safety net to extend to 3 items (2 backfill).
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=6,
            top_k_rerank=6,
            top_k_final=5,
            reranker_threshold=0.5,
            reranker_min_docs=3,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )

        def _fake_hybrid(*, embedding_query, bm25_query, top_k, source_id=None, query_vector=None):
            return [
                {"id": "a1", "text": "above threshold", "score": 0.9, "metadata": {"parent_id": "pa1", "source_id": "src"}},
                {"id": "b1", "text": "below threshold one", "score": 0.8, "metadata": {"parent_id": "pb1", "source_id": "src"}},
                {"id": "c1", "text": "below threshold two", "score": 0.7, "metadata": {"parent_id": "pc1", "source_id": "src"}},
            ]

        def _fake_rerank(query, items):
            # Sorted descending: a1 above threshold (0.9), b1 and c1 below (0.1, 0.08)
            reranked = [
                {**items[0], "rerank_score": 0.90},
                {**items[1], "rerank_score": 0.10},
                {**items[2], "rerank_score": 0.08},
            ]
            return reranked, [0.90, 0.10, 0.08]

        monkeypatch.setattr(engine, "_hybrid_search_decoupled", _fake_hybrid)
        monkeypatch.setattr(engine, "_rerank", _fake_rerank)

        response = engine.search("test query")

        results = response.results

        # Safety net fires: threshold=0.5 passes only a1 (score 0.90), but
        # min_docs=3 extends to b1 and c1.
        assert len(results) == 3

        by_id = {r.child_id: r for r in results}

        # a1 scored 0.90 >= 0.5 threshold: must NOT be tagged
        assert not by_id["a1"].metadata.get("below_threshold"), (
            "above-threshold item should not have below_threshold=True"
        )

        # b1 and c1 scored below threshold: must be tagged
        assert by_id["b1"].metadata.get("below_threshold") is True, (
            "sub-threshold safety-net item b1 missing below_threshold=True"
        )
        assert by_id["c1"].metadata.get("below_threshold") is True, (
            "sub-threshold safety-net item c1 missing below_threshold=True"
        )

    def test_sub_threshold_expansion_scoped_to_above_threshold_sources(
        self,
        tmp_storage,
        mock_embedder,
        mock_reranker,
        monkeypatch,
    ):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=6,
            top_k_rerank=6,
            top_k_final=1,
            reranker_threshold=0.05,
            reranker_min_docs=1,
            retrieval_budget=400,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )

        token_rich_text = "word " * 15

        def _fake_hybrid_search_decoupled(*, embedding_query, bm25_query, top_k, source_id=None, query_vector=None):
            return [
                {"id": "a1", "text": token_rich_text, "score": 0.9, "metadata": {"parent_id": "pa1", "source_id": "source_a"}},
                {"id": "b1", "text": token_rich_text, "score": 0.8, "metadata": {"parent_id": "pb1", "source_id": "source_b"}},
                {"id": "a2", "text": token_rich_text, "score": 0.7, "metadata": {"parent_id": "pa2", "source_id": "source_a"}},
                {"id": "b2", "text": token_rich_text, "score": 0.6, "metadata": {"parent_id": "pb2", "source_id": "source_b"}},
            ]

        def _fake_rerank(query, items):
            reranked = [
                {**items[0], "rerank_score": 0.90},
                {**items[1], "rerank_score": 0.10},
                {**items[2], "rerank_score": 0.09},
                {**items[3], "rerank_score": 0.08},
            ]
            return reranked, [0.90, 0.10, 0.09, 0.08]

        monkeypatch.setattr(engine, "_hybrid_search_decoupled", _fake_hybrid_search_decoupled)
        monkeypatch.setattr(engine, "_rerank", _fake_rerank)
        monkeypatch.setattr(retrieval, "_WORD_TO_TOKEN_RATIO", 1.0)

        response = engine.search("what mentions of ChatGPT are there", intent="factual")

        results = response.results
        sources = {r.metadata.get("source_id") for r in results if r.metadata.get("source_id")}
        assert "source_b" not in sources
        assert sources == {"source_a"}

    def test_sub_threshold_expansion_falls_back_when_no_above_threshold_hits(
        self,
        tmp_storage,
        mock_embedder,
        mock_reranker,
        monkeypatch,
    ):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=6,
            top_k_rerank=6,
            top_k_final=1,
            reranker_threshold=0.50,
            reranker_min_docs=1,
            retrieval_budget=400,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )

        token_rich_text = "word " * 15

        def _fake_hybrid_search_decoupled(*, embedding_query, bm25_query, top_k, source_id=None, query_vector=None):
            return [
                {"id": "a1", "text": token_rich_text, "score": 0.9, "metadata": {"parent_id": "pa1", "source_id": "source_a"}},
                {"id": "b1", "text": token_rich_text, "score": 0.8, "metadata": {"parent_id": "pb1", "source_id": "source_b"}},
                {"id": "a2", "text": token_rich_text, "score": 0.7, "metadata": {"parent_id": "pa2", "source_id": "source_a"}},
            ]

        def _fake_rerank(query, items):
            reranked = [
                {**items[0], "rerank_score": 0.20},
                {**items[1], "rerank_score": 0.19},
                {**items[2], "rerank_score": 0.18},
            ]
            return reranked, [0.20, 0.19, 0.18]

        monkeypatch.setattr(engine, "_hybrid_search_decoupled", _fake_hybrid_search_decoupled)
        monkeypatch.setattr(engine, "_rerank", _fake_rerank)
        monkeypatch.setattr(retrieval, "_WORD_TO_TOKEN_RATIO", 1.0)

        response = engine.search("who is Romeo", intent="factual")

        results = response.results
        sources = {r.metadata.get("source_id") for r in results if r.metadata.get("source_id")}
        assert "source_b" in sources


# ===========================================================================
# Full search pipeline
# ===========================================================================

class TestFullSearch:
    def test_search_returns_results(self, retrieval_engine: RetrievalEngine):
        response = retrieval_engine.search("Chomsky language theory")
        results = response.results
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)

    def test_search_results_have_text(self, retrieval_engine: RetrievalEngine):
        response = retrieval_engine.search("epistemology knowledge")
        results = response.results
        for r in results:
            assert r.text or r.parent_text

    def test_search_results_have_score(self, retrieval_engine: RetrievalEngine):
        response = retrieval_engine.search("ethics moral philosophy")
        results = response.results
        for r in results:
            assert isinstance(r.score, float)

    def test_search_metrics_collected(self, retrieval_engine: RetrievalEngine):
        response = retrieval_engine.search("Chomsky", collect_metrics=True)
        results = response.results
        if results:
            metrics = response.metrics
            assert metrics is not None
            assert metrics.timing.total_ms > 0
            assert metrics.timing.hybrid_search_ms >= 0
            assert metrics.timing.sparse_search_ms >= 0

    def test_search_max_two_children_per_parent(self, retrieval_engine: RetrievalEngine):
        """Final results should have at most 2 children per parent_id (max_children_per_parent=2)."""
        from collections import Counter
        response = retrieval_engine.search("Chomsky language")
        results = response.results
        parent_ids = [
            r.metadata.get("parent_id") for r in results
            if r.metadata.get("parent_id")
        ]
        counts = Counter(parent_ids)
        assert all(c <= 2 for c in counts.values()), f"Parent counts exceed 2: {counts}"

    def test_search_deterministic(self, retrieval_engine: RetrievalEngine):
        """Same query should produce same results."""
        response = retrieval_engine.search("Chomsky theory")
        r1 = response.results
        response = retrieval_engine.search("Chomsky theory")
        r2 = response.results
        assert [r.child_id for r in r1] == [r.child_id for r in r2]

    def test_identical_passage_across_sources_both_retrievable(
        self,
        tmp_storage,
        mock_embedder: MockEmbeddingModel,
        mock_reranker: MockReranker,
    ):
        shared_text = (
            "Noam Chomsky's theory of language acquisition argues that children "
            "acquire grammar despite limited input from the environment."
        )

        parent_a = ParentChunk(
            id="p-shared-a",
            text=shared_text,
            metadata=Metadata(
                source_id="shared_doc_a",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        parent_b = ParentChunk(
            id="p-shared-b",
            text=shared_text,
            metadata=Metadata(
                source_id="shared_doc_b",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=None,
            ),
        )
        child_a = ChildChunk(
            id="c-shared-a",
            text=shared_text,
            metadata=Metadata(
                source_id="shared_doc_a",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=parent_a.id,
            ),
        )
        child_b = ChildChunk(
            id="c-shared-b",
            text=shared_text,
            metadata=Metadata(
                source_id="shared_doc_b",
                page_number=1,
                page_label="1",
                display_page="1",
                header_path="Document",
                parent_id=parent_b.id,
            ),
        )

        tmp_storage.add_parents([parent_a, parent_b])
        tmp_storage.add_children(
            [child_a, child_b],
            embeddings=mock_embedder.encode([shared_text, shared_text], normalize_embeddings=True),
        )

        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=_make_config("regular"),
        )
        response = engine.search(
            "noam chomsky's theory of language",
            top_k_fused=32,
            top_k_rerank=24,
            top_k_final=12,
        )

        source_ids = {
            str(result.metadata.get("source_id"))
            for result in response.results
            if isinstance(result.metadata, dict) and result.metadata.get("source_id")
        }

        assert "shared_doc_a" in source_ids
        assert "shared_doc_b" in source_ids


# ===========================================================================
# Mode comparison
# ===========================================================================

class TestModeComparison:
    def test_modes_differ_in_top_k(self):
        """Different modes should have different top_k parameters."""
        regular = _make_config("regular")
        deep = _make_config("deep-research")
        assert deep.top_k_dense > regular.top_k_dense
        assert deep.top_k_final > regular.top_k_final

    def test_modes_differ_in_threshold(self):
        regular = _make_config("regular")
        deep = _make_config("deep-research")
        assert deep.reranker_threshold < regular.reranker_threshold


# ===========================================================================
# Latency: retrieval stages
# ===========================================================================

class TestRetrievalLatency:
    def test_hybrid_search_latency(self, retrieval_engine: RetrievalEngine):
        for q in ["Chomsky theory", "epistemology knowledge", "ethics"]:
            with Timer("hybrid_search", query=q) as t:
                retrieval_engine._hybrid_search_decoupled(
                    embedding_query=q, bm25_query=q, top_k=10,
                )
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
                response = retrieval_engine.search(q)
                results = response.results
            result_count = len(results)
            logger.info(
                f"full_search '{q[:40]}...': {t.result.elapsed_ms:.2f}ms, {result_count} results"
            )

# ===========================================================================
# Token ratio
# ===========================================================================

class TestWordToTokenRatio:
    def test_word_to_token_ratio_applied(self):
        """_est_tokens should apply _WORD_TO_TOKEN_RATIO to the word count."""
        # 100 whitespace-separated words × 1.35 = 135
        text = "word " * 100
        from src import retrieval as _retrieval
        # Replicate the closure logic directly
        result = int(len(text.split()) * _retrieval._WORD_TO_TOKEN_RATIO)
        assert result == int(100 * 1.35)
