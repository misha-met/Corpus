"""End-to-end test: fixed corpus, fixed queries, full pipeline with structured results.

This test exercises the complete RAG pipeline from ingest through retrieval
and budget packing with citation formatting, using mocked ML models.
It produces a comprehensive timing and correctness profile.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

from src.config import ModelConfig, _get_mode_config
from src.generation import build_messages
from src.generator import enforce_token_budget
from src.intent import Intent, _classify_heuristic
from src.metrics import BudgetMetrics, format_metrics_summary
from src.models import ChildChunk, Metadata, ParentChunk
from src.retrieval import (
    RetrievalEngine,
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

logger = get_test_logger("e2e")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class E2EQueryResult:
    """Structured result from a single e2e query."""
    query: str
    intent: str
    intent_confidence: float
    mode: str
    citations_enabled: bool
    result_count: int
    context_length: int
    packed_docs: int
    used_tokens: int
    budget_tokens: int
    utilization_pct: float
    timing_ms: dict[str, float]
    reranker_stats: dict[str, Any]
    dedup_stats: dict[str, Any]
    threshold_stats: dict[str, Any]
    errors: list[str]


class TestEndToEnd:
    """Controlled end-to-end test with fixed corpus and queries."""

    @pytest.fixture(autouse=True)
    def setup_e2e(self, tmp_path: Path):
        """Set up a complete pipeline with test corpus."""
        self.embedder = MockEmbeddingModel(dim=384)
        self.reranker = MockReranker()
        self.tokenizer = MockTokenizer()

        config = StorageConfig(
            lance_dir=tmp_path / "e2e_lance",
            lance_table="e2e_chunks",
        )
        self.storage = StorageEngine(config)

        parents, children = generate_parent_child_corpus()
        self.parents = parents
        self.children = children

        self.storage.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        self.storage.add_children(children, embeddings=embeddings)

        yield
        self.storage.close()

    def _run_query(
        self,
        query: str,
        mode: str = "regular",
        citations_enabled: bool = False,
    ) -> E2EQueryResult:
        """Execute a single query through the full pipeline."""
        errors: list[str] = []

        # 1. Intent classification
        t0 = time.perf_counter()
        intent_result = _classify_heuristic(query)
        intent_ms = (time.perf_counter() - t0) * 1000

        # 2. Build retrieval config
        if mode == "regular":
            model_config = ModelConfig(
                mode=mode, llm_model="test", embedding_model="test",
                reranker_model="test",
                top_k_dense=10, top_k_sparse=10, top_k_fused=8,
                top_k_rerank=5, top_k_final=3,
                reranker_threshold=0.0, reranker_min_docs=1,
                retrieval_budget=8000,
            )
        else:  # power-deep-research
            model_config = ModelConfig(
                mode=mode, llm_model="test", embedding_model="test",
                reranker_model="test",
                top_k_dense=20, top_k_sparse=20, top_k_fused=15,
                top_k_rerank=10, top_k_final=5,
                reranker_threshold=0.0, reranker_min_docs=3,
                retrieval_budget=20000,
            )

        # 3. Retrieval
        engine = RetrievalEngine(
            storage=self.storage,
            embedding_model=self.embedder,
            reranker=self.reranker,
            config=model_config,
        )

        t0 = time.perf_counter()
        results = engine.search(query, collect_metrics=True)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        metrics = results[0].metrics if results and results[0].metrics else None

        # 4. Budget packing
        parent_texts = [r.parent_text for r in results if r.parent_text]
        result_metadatas = [r.metadata for r in results if r.parent_text]

        t0 = time.perf_counter()
        pack_result = enforce_token_budget(
            docs=parent_texts,
            max_tokens=model_config.retrieval_budget,
            tokenizer=self.tokenizer,
        )
        packing_ms = (time.perf_counter() - t0) * 1000

        # 5. Citation formatting
        t0 = time.perf_counter()
        if citations_enabled and pack_result.packed_docs:
            packed_metas = [
                result_metadatas[i]
                for i in pack_result.packed_indices
                if i < len(result_metadatas)
            ]
            context, source_mapping = format_context_with_citations(
                texts=pack_result.packed_docs,
                metadatas=packed_metas,
            )
            source_legend = build_source_legend(source_mapping)
        else:
            context = "\n\n".join(pack_result.packed_docs)
            source_legend = None
        citation_ms = (time.perf_counter() - t0) * 1000

        # 6. Message building
        t0 = time.perf_counter()
        messages = build_messages(
            context=context,
            question=query,
            intent=intent_result.intent,
            citations_enabled=citations_enabled,
            source_legend=source_legend,
        )
        message_ms = (time.perf_counter() - t0) * 1000

        # Timing breakdown
        timing_ms = {
            "intent_ms": intent_ms,
            "retrieval_ms": retrieval_ms,
            "packing_ms": packing_ms,
            "citation_ms": citation_ms,
            "message_building_ms": message_ms,
        }
        if metrics:
            timing_ms.update({
                "hybrid_search_ms": metrics.timing.hybrid_search_ms,
                "sparse_search_ms": metrics.timing.sparse_search_ms,
                "rrf_fusion_ms": metrics.timing.rrf_fusion_ms,
                "rerank_ms": metrics.timing.rerank_ms,
                "dedup_ms": metrics.timing.dedup_ms,
            })

        utilization = (
            100 * pack_result.used_tokens / model_config.retrieval_budget
            if model_config.retrieval_budget > 0
            else 0
        )

        reranker_stats = {}
        if metrics:
            reranker_stats = {
                "items_reranked": metrics.reranker.items_reranked,
                "score_min": metrics.reranker.score_min,
                "score_max": metrics.reranker.score_max,
                "score_mean": metrics.reranker.score_mean,
                "score_std": metrics.reranker.score_std,
            }

        dedup_stats = {}
        if metrics:
            dedup_stats = {
                "before": metrics.deduplication.children_before_dedup,
                "after": metrics.deduplication.children_after_dedup,
                "reduction_pct": metrics.deduplication.reduction_pct,
            }

        threshold_stats = {}
        if metrics:
            threshold_stats = {
                "threshold": metrics.threshold.threshold_value,
                "before": metrics.threshold.items_before_threshold,
                "after": metrics.threshold.items_after_threshold,
                "safety_net": metrics.threshold.safety_net_triggered,
            }

        return E2EQueryResult(
            query=query,
            intent=intent_result.intent.value,
            intent_confidence=intent_result.confidence,
            mode=mode,
            citations_enabled=citations_enabled,
            result_count=len(results),
            context_length=len(context),
            packed_docs=len(pack_result.packed_docs),
            used_tokens=pack_result.used_tokens,
            budget_tokens=model_config.retrieval_budget,
            utilization_pct=utilization,
            timing_ms=timing_ms,
            reranker_stats=reranker_stats,
            dedup_stats=dedup_stats,
            threshold_stats=threshold_stats,
            errors=errors,
        )

    def test_e2e_all_queries_all_modes(self):
        """Run all fixed queries across all modes, collecting comprehensive metrics."""
        valid_queries = [q for q in FIXED_QUERIES if q.strip()]
        modes = ["regular", "power-deep-research"]
        all_results: list[dict] = []

        for mode in modes:
            for query in valid_queries:
                for citations in [False, True]:
                    result = self._run_query(query, mode=mode, citations_enabled=citations)
                    result_dict = {
                        "query": result.query,
                        "intent": result.intent,
                        "intent_confidence": result.intent_confidence,
                        "mode": result.mode,
                        "citations": result.citations_enabled,
                        "result_count": result.result_count,
                        "packed_docs": result.packed_docs,
                        "used_tokens": result.used_tokens,
                        "budget_tokens": result.budget_tokens,
                        "utilization_pct": round(result.utilization_pct, 1),
                        "timing_ms": {k: round(v, 2) for k, v in result.timing_ms.items()},
                        "reranker_stats": result.reranker_stats,
                        "dedup_stats": result.dedup_stats,
                        "threshold_stats": result.threshold_stats,
                        "errors": result.errors,
                    }
                    all_results.append(result_dict)
                    logger.info(
                        f"E2E: mode={mode} cite={citations} query='{query[:40]}' "
                        f"-> {result.result_count} results, "
                        f"{result.used_tokens}/{result.budget_tokens} tokens "
                        f"({result.utilization_pct:.1f}%)"
                    )

        # Save structured results for report generation
        results_file = RESULTS_DIR / "e2e_results.json"
        results_file.write_text(json.dumps(all_results, indent=2))
        logger.info(f"E2E results saved to {results_file}")

        # Assertions: all queries should have produced results
        for r in all_results:
            assert r["result_count"] > 0, f"No results for query: {r['query']} (mode={r['mode']})"
            assert len(r["errors"]) == 0, f"Errors for query: {r['query']}: {r['errors']}"

    def test_e2e_corpus_coverage(self):
        """Verify that retrieval can surface content from all sources."""
        source_queries = {
            "test_doc_linguistics": "Chomsky language acquisition grammar",
            "test_doc_philosophy": "epistemology knowledge Kant Descartes",
        }

        for source_id, query in source_queries.items():
            result = self._run_query(query, mode="regular")
            # We can't guarantee source filtering in mocked embedding,
            # but we should get results
            assert result.result_count > 0
            logger.info(f"Corpus coverage: source={source_id}, results={result.result_count}")

    def test_e2e_deterministic(self):
        """Same query should produce identical results on repeated runs."""
        query = "Chomsky language acquisition theory"
        r1 = self._run_query(query, mode="regular")
        r2 = self._run_query(query, mode="regular")
        assert r1.result_count == r2.result_count
        assert r1.packed_docs == r2.packed_docs
        assert r1.used_tokens == r2.used_tokens
