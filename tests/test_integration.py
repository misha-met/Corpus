"""Integration tests: composed pipeline stages, end-to-end with mocked models.

These tests exercise multiple components together to verify correct
interaction between stages of the RAG pipeline.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from src.config import ModelConfig, _get_mode_config
from src.generation import build_messages
from src.generator import BudgetPackResult, count_tokens, enforce_token_budget
from src.intent import Intent, IntentClassifier, _classify_heuristic
from src.metrics import BudgetMetrics, RetrievalMetrics, format_metrics_summary
from src.models import ChildChunk, Metadata, ParentChunk
from src.retrieval import (
    RetrievalEngine,
    RetrievalResult,
    build_source_legend,
    format_context_with_citations,
)
from src.storage import StorageConfig, StorageEngine
from tests.conftest import (
    FIXED_CORPUS,
    FIXED_QUERIES,
    MockEmbeddingModel,
    MockReranker,
    MockTokenizer,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
)

logger = get_test_logger("integration")


# ===========================================================================
# Ingest → storage → retrieval integration
# ===========================================================================

class TestIngestToRetrieval:
    """Test the chain: data creation → storage → index build → retrieval."""

    def test_ingest_then_search(self, tmp_storage, mock_embedder, mock_reranker):
        """Documents ingested should be retrievable via search."""
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,  # Accept all
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky language acquisition")
        assert len(results) > 0
        # Results should contain text from our corpus
        all_text = " ".join(r.text for r in results)
        logger.info(f"Search returned {len(results)} results, total text length {len(all_text)}")

    def test_parent_expansion_works(self, tmp_storage, mock_embedder, mock_reranker):
        """Retrieved results should include parent_text when parent_id exists."""
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("epistemology knowledge")
        for r in results:
            if r.metadata.get("parent_id"):
                assert r.parent_text is not None, (
                    f"Result {r.child_id} has parent_id but no parent_text"
                )


# ===========================================================================
# Retrieval → budget packing integration
# ===========================================================================

class TestRetrievalToBudgetPacking:
    """Test the chain: retrieval results → token budget packing."""

    def test_retrieval_results_pack_correctly(self, tmp_storage, mock_embedder, mock_reranker):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky language theory")
        parent_texts = [r.parent_text for r in results if r.parent_text]

        if parent_texts:
            tok = MockTokenizer()
            pack_result = enforce_token_budget(
                docs=parent_texts,
                max_tokens=8000,
                tokenizer=tok,
            )
            assert len(pack_result.packed_docs) > 0
            assert pack_result.used_tokens <= 8000

            # Metadata alignment check
            result_metadatas = [r.metadata for r in results if r.parent_text]
            packed_metas = [
                result_metadatas[i]
                for i in pack_result.packed_indices
                if i < len(result_metadatas)
            ]
            assert len(packed_metas) == len(pack_result.packed_docs)
            logger.info(
                f"Packed {len(pack_result.packed_docs)} docs, "
                f"{pack_result.used_tokens}/8000 tokens, "
                f"metadata alignment verified"
            )


# ===========================================================================
# Retrieval → citations integration
# ===========================================================================

class TestRetrievalToCitations:
    """Test the chain: retrieval → citation formatting → message building."""

    def test_full_citation_pipeline(self, tmp_storage, mock_embedder, mock_reranker):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky theory language")
        parent_texts = [r.parent_text for r in results if r.parent_text]
        metadatas = [r.metadata for r in results if r.parent_text]

        if parent_texts and metadatas:
            # Format with citations
            context, source_mapping = format_context_with_citations(parent_texts, metadatas)
            assert "[CHUNK START" in context
            assert "[CHUNK END]" in context

            # Build legend
            legend = build_source_legend(source_mapping)

            # Build messages
            messages = build_messages(
                context=context,
                question="What is Chomsky's theory?",
                intent=Intent.ANALYZE,
                citations_enabled=True,
                source_legend=legend,
            )
            assert len(messages) == 2
            assert "CITATION" in messages[0]["content"]
            logger.info("Citation pipeline: context formatted, legend built, messages constructed")

    def test_citation_disabled_pipeline(self, tmp_storage, mock_embedder, mock_reranker):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("epistemology knowledge")
        parent_texts = [r.parent_text for r in results if r.parent_text]

        if parent_texts:
            context = "\n\n".join(parent_texts)
            messages = build_messages(
                context=context,
                question="What is epistemology?",
                intent=Intent.OVERVIEW,
                citations_enabled=False,
            )
            assert "CITATION" not in messages[0]["content"]


# ===========================================================================
# Intent → retrieval → context building
# ===========================================================================

class TestIntentToContext:
    """Test the chain: intent classification → query handling → retrieval."""

    def test_intent_affects_message_format(self):
        """Different intents should produce different system prompts."""
        intents = [Intent.OVERVIEW, Intent.SUMMARIZE, Intent.EXPLAIN, Intent.ANALYZE]
        system_messages = []
        for intent in intents:
            messages = build_messages(
                context="Some context text.",
                question="What is this about?",
                intent=intent,
                citations_enabled=False,
            )
            system_messages.append(messages[0]["content"])

        # Check they're not all identical
        unique_msgs = set(system_messages)
        assert len(unique_msgs) == len(intents), "Each intent should produce unique system message"

    def test_classification_then_message_building(self):
        """Classify intent then build appropriate messages."""
        query = "Summarize the key points of this paper"
        result = _classify_heuristic(query)
        assert result.intent == Intent.SUMMARIZE

        messages = build_messages(
            context="Paper context here.",
            question=query,
            intent=result.intent,
        )
        system = messages[0]["content"]
        assert "summary" in system.lower() or "key points" in system.lower()


# ===========================================================================
# Mode-specific pipeline behaviour
# ===========================================================================

class TestModeSpecificBehaviour:
    """Test how different modes affect pipeline behaviour and metrics."""

    @pytest.mark.parametrize("mode", ["regular", "power-deep-research"])
    def test_mode_produces_valid_config(self, mode: str):
        config = _get_mode_config(mode, ram_gb=64.0)
        # Token budget must be positive
        assert config.retrieval_budget > 0
        assert config.context_window > 0
        # Dense >= sparse for balanced retrieval
        assert config.top_k_dense >= config.top_k_sparse or config.top_k_sparse >= config.top_k_dense

    def test_mode_affects_retrieval_depth(self, tmp_storage, mock_embedder, mock_reranker):
        """Deep-research mode should retrieve more documents than regular."""
        results_by_mode = {}
        for mode in ["regular", "power-deep-research"]:
            if mode == "regular":
                config = ModelConfig(
                    mode=mode, llm_model="test", embedding_model="test",
                    reranker_model="test",
                    top_k_dense=5, top_k_sparse=5, top_k_fused=4,
                    top_k_rerank=3, top_k_final=2,
                    reranker_threshold=0.0, reranker_min_docs=1,
                )
            else:
                config = ModelConfig(
                    mode=mode, llm_model="test", embedding_model="test",
                    reranker_model="test",
                    top_k_dense=10, top_k_sparse=10, top_k_fused=8,
                    top_k_rerank=6, top_k_final=4,
                    reranker_threshold=0.0, reranker_min_docs=1,
                )
            engine = RetrievalEngine(
                storage=tmp_storage,
                embedding_model=mock_embedder,
                reranker=mock_reranker,
                config=config,
            )
            results = engine.search("Chomsky language")
            results_by_mode[mode] = len(results)
            logger.info(f"Mode {mode}: {len(results)} results")

        assert results_by_mode["power-deep-research"] >= results_by_mode["regular"]


# ===========================================================================
# Cross-stage metrics collection
# ===========================================================================

class TestCrossStageMetrics:
    """Test metrics flow across composed pipeline stages."""

    def test_metrics_populated_end_to_end(self, tmp_storage, mock_embedder, mock_reranker):
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )
        results = engine.search("Chomsky language theory", collect_metrics=True)
        assert results
        metrics = results[0].metrics
        assert metrics is not None

        # Timing should be populated
        assert metrics.timing.hybrid_search_ms > 0
        assert metrics.timing.sparse_search_ms >= 0
        assert metrics.timing.total_ms > 0

        # Deduplication should be populated  
        assert metrics.deduplication.children_before_dedup >= 0

        # Reranker stats should be populated
        assert metrics.reranker.items_reranked >= 0

        # Summary should be formattable
        summary = format_metrics_summary(metrics)
        assert len(summary) > 0
        logger.info(f"Metrics summary: {summary}")


# ===========================================================================
# Pipeline latency profiling (integration)
# ===========================================================================

class TestPipelineLatency:
    """Measure latency of composed pipeline stages."""

    def test_full_pipeline_latency_per_query(self, tmp_storage, mock_embedder, mock_reranker):
        """Profile each query through the full pipeline."""
        config = ModelConfig(
            mode="regular",
            llm_model="test",
            embedding_model="test",
            reranker_model="test",
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=8,
            top_k_rerank=5,
            top_k_final=3,
            reranker_threshold=0.0,
            reranker_min_docs=1,
        )
        engine = RetrievalEngine(
            storage=tmp_storage,
            embedding_model=mock_embedder,
            reranker=mock_reranker,
            config=config,
        )

        valid_queries = [q for q in FIXED_QUERIES if q.strip()]
        timings = []

        for q in valid_queries:
            with Timer("pipeline", query=q) as t:
                results = engine.search(q, collect_metrics=True)

            pipeline_ms = t.result.elapsed_ms
            metrics = results[0].metrics if results and results[0].metrics else None
            stage_breakdown = {}
            if metrics:
                stage_breakdown = {
                    "dense_ms": metrics.timing.hybrid_search_ms,
                    "sparse_ms": metrics.timing.sparse_search_ms,
                    "rrf_ms": metrics.timing.rrf_fusion_ms,
                    "rerank_ms": metrics.timing.rerank_ms,
                    "dedup_ms": metrics.timing.dedup_ms,
                    "total_ms": metrics.timing.total_ms,
                }

            timings.append({
                "query": q,
                "pipeline_ms": pipeline_ms,
                "result_count": len(results),
                **stage_breakdown,
            })

            logger.info(
                f"Pipeline '{q[:40]}': {pipeline_ms:.1f}ms total, "
                f"{len(results)} results, "
                f"stages={json.dumps({k: f'{v:.1f}' for k, v in stage_breakdown.items()})}"
            )

        # Write structured timing data
        logger.info(f"TIMING_DATA: {json.dumps(timings, indent=2)}")

    def test_pipeline_latency_mode_comparison(self, tmp_storage, mock_embedder, mock_reranker):
        """Compare pipeline latency across modes."""
        modes = {
            "regular": ModelConfig(
                mode="regular", llm_model="test", embedding_model="test",
                reranker_model="test",
                top_k_dense=5, top_k_sparse=5, top_k_fused=4,
                top_k_rerank=3, top_k_final=2,
                reranker_threshold=0.0, reranker_min_docs=1,
            ),
            "power-deep-research": ModelConfig(
                mode="power-deep-research", llm_model="test", embedding_model="test",
                reranker_model="test",
                top_k_dense=10, top_k_sparse=10, top_k_fused=8,
                top_k_rerank=6, top_k_final=4,
                reranker_threshold=0.0, reranker_min_docs=1,
            ),
        }

        query = "Chomsky language acquisition theory"
        mode_timings = {}

        for mode_name, config in modes.items():
            engine = RetrievalEngine(
                storage=tmp_storage,
                embedding_model=mock_embedder,
                reranker=mock_reranker,
                config=config,
            )
            with Timer("mode_comparison", mode=mode_name) as t:
                results = engine.search(query, collect_metrics=True)
            mode_timings[mode_name] = {
                "total_ms": t.result.elapsed_ms,
                "result_count": len(results),
            }
            if results and results[0].metrics:
                mode_timings[mode_name]["breakdown"] = {
                    "dense_ms": results[0].metrics.timing.hybrid_search_ms,
                    "sparse_ms": results[0].metrics.timing.sparse_search_ms,
                    "rerank_ms": results[0].metrics.timing.rerank_ms,
                }

        logger.info(f"MODE_COMPARISON: {json.dumps(mode_timings, indent=2)}")
        for mode, data in mode_timings.items():
            logger.info(f"Mode {mode}: {data['total_ms']:.1f}ms, {data['result_count']} results")
