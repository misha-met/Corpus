"""Latency profiler: systematic performance measurement across pipeline stages.

This module provides a structured latency profiling framework that measures
each stage of the RAG pipeline independently and reports results in a format
suitable for analysis and comparison.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.config import ModelConfig
from src.generator import enforce_token_budget
from src.retrieval import RetrievalEngine
from src.storage import StorageConfig, StorageEngine
from tests.conftest import (
    FIXED_QUERIES,
    MockEmbeddingModel,
    MockReranker,
    MockTokenizer,
    Timer,
    generate_parent_child_corpus,
    get_test_logger,
)

logger = get_test_logger("latency_profiler")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class LatencyProfile:
    """Profile for a single operation across multiple iterations."""
    label: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(min(self.times_ms), 3) if self.times_ms else 0,
            "max_ms": round(max(self.times_ms), 3) if self.times_ms else 0,
            "context": self.context,
        }


class TestLatencyProfiler:
    """Systematic latency profiling of each pipeline stage."""

    ITERATIONS = 5  # Repeat each measurement for stability

    @pytest.fixture(autouse=True)
    def setup_profiler(self, tmp_path: Path):
        self.embedder = MockEmbeddingModel(dim=384)
        self.reranker = MockReranker()
        self.tokenizer = MockTokenizer()

        config = StorageConfig(
            sqlite_path=tmp_path / "perf.sqlite",
            chroma_dir=tmp_path / "perf_chroma",
            chroma_collection="perf_chunks",
        )
        self.storage = StorageEngine(config)

        parents, children = generate_parent_child_corpus()
        self.storage.add_parents(parents)
        texts = [c.text for c in children]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        self.storage.add_children(children, embeddings=embeddings)

        bm25_path = tmp_path / "bm25.json"
        self.storage.persist_bm25(bm25_path)
        self.storage.load_bm25(bm25_path)

        self.model_config = ModelConfig(
            mode="regular", llm_model="test", embedding_model="test",
            reranker_model="test",
            top_k_dense=10, top_k_sparse=10, top_k_fused=8,
            top_k_rerank=5, top_k_final=3,
            reranker_threshold=0.0, reranker_min_docs=1,
        )

        yield
        self.storage.close()

    def _profile(self, label: str, fn, iterations: int = None, **context) -> LatencyProfile:
        """Run a function multiple times and collect timing data."""
        n = iterations or self.ITERATIONS
        profile = LatencyProfile(label=label, iterations=n, context=context)
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - t0) * 1000
            profile.times_ms.append(elapsed)
        return profile

    def test_profile_embedding_encode(self):
        """Profile embedding model encode latency."""
        texts = ["Chomsky theory of language acquisition", "epistemology and knowledge"]
        profile = self._profile(
            "embedding_encode",
            lambda: self.embedder.encode(texts, normalize_embeddings=True),
            n_texts=len(texts),
        )
        logger.info(f"embedding_encode: {json.dumps(profile.to_dict())}")

    def test_profile_vector_query(self):
        """Profile vector (Chroma) query latency."""
        q_emb = self.embedder.encode(["test query"], normalize_embeddings=True)
        for k in [5, 10]:
            profile = self._profile(
                f"vector_query_k{k}",
                lambda: self.storage.query_children(embeddings=q_emb, top_k=k),
                top_k=k,
            )
            logger.info(f"vector_query_k{k}: {json.dumps(profile.to_dict())}")

    def test_profile_bm25_search(self):
        """Profile BM25 search latency."""
        bm25 = self.storage.bm25
        for query in ["Chomsky language", "epistemology knowledge Kant"]:
            tokens = query.split()
            profile = self._profile(
                f"bm25_search",
                lambda: bm25.get_scores(tokens),
                query=query,
            )
            logger.info(f"bm25_search '{query}': {json.dumps(profile.to_dict())}")

    def test_profile_rrf_fusion(self):
        """Profile RRF fusion latency with varying input sizes."""
        for n in [10, 50, 100]:
            dense = [{"id": f"d{i}", "rank": i + 1} for i in range(n)]
            sparse = [{"id": f"s{i}", "rank": i + 1} for i in range(n)]
            profile = self._profile(
                f"rrf_fusion_{n}",
                lambda: RetrievalEngine._rrf_fuse(dense, sparse),
                input_size=n,
            )
            logger.info(f"rrf_fusion n={n}: {json.dumps(profile.to_dict())}")

    def test_profile_deduplication(self):
        """Profile deduplication latency with varying input sizes."""
        for n in [20, 50, 100]:
            items = [
                {"id": f"c{i}", "score": float(i), "metadata": {"parent_id": f"p{i % (n // 3)}"}}
                for i in range(n)
            ]
            profile = self._profile(
                f"dedup_{n}",
                lambda: RetrievalEngine._deduplicate_by_parent(items, top_k=n),
                input_size=n,
            )
            logger.info(f"dedup n={n}: {json.dumps(profile.to_dict())}")

    def test_profile_reranking(self):
        """Profile reranking latency with varying batch sizes."""
        for batch in [5, 10, 20]:
            items = [
                {"id": f"c{i}", "text": f"document about topic {i} with some content words"}
                for i in range(batch)
            ]
            engine = RetrievalEngine(
                storage=self.storage,
                embedding_model=self.embedder,
                reranker=self.reranker,
                config=self.model_config,
            )
            profile = self._profile(
                f"rerank_{batch}",
                lambda: engine._rerank("test query topic", items),
                batch_size=batch,
            )
            logger.info(f"rerank batch={batch}: {json.dumps(profile.to_dict())}")

    def test_profile_budget_packing(self):
        """Profile token budget packing latency with varying doc counts."""
        for n_docs in [5, 20, 50, 100]:
            docs = [f"document content about topic {i} " * 20 for i in range(n_docs)]
            profile = self._profile(
                f"budget_packing_{n_docs}",
                lambda: enforce_token_budget(docs, max_tokens=5000, tokenizer=self.tokenizer),
                n_docs=n_docs,
            )
            logger.info(f"budget_packing n={n_docs}: {json.dumps(profile.to_dict())}")

    def test_profile_parent_lookup(self):
        """Profile SQLite parent text lookup latency."""
        parents, _ = generate_parent_child_corpus()
        parent_id = parents[0].id
        profile = self._profile(
            "parent_lookup",
            lambda: self.storage.get_parent_text(parent_id),
            iterations=20,
        )
        logger.info(f"parent_lookup: {json.dumps(profile.to_dict())}")

    def test_profile_full_pipeline(self):
        """Profile the complete search pipeline per query."""
        engine = RetrievalEngine(
            storage=self.storage,
            embedding_model=self.embedder,
            reranker=self.reranker,
            config=self.model_config,
        )

        all_profiles = []
        valid_queries = [q for q in FIXED_QUERIES if q.strip()]

        for query in valid_queries:
            profile = self._profile(
                "full_pipeline",
                lambda: engine.search(query, collect_metrics=True),
                query=query,
            )
            all_profiles.append(profile.to_dict())
            logger.info(f"full_pipeline '{query[:40]}': {json.dumps(profile.to_dict())}")

        # Save profiling results
        results_file = RESULTS_DIR / "latency_profiles.json"
        results_file.write_text(json.dumps(all_profiles, indent=2))
        logger.info(f"Latency profiles saved to {results_file}")

    def test_profile_summary_report(self):
        """Generate a summary of all profiling data."""
        engine = RetrievalEngine(
            storage=self.storage,
            embedding_model=self.embedder,
            reranker=self.reranker,
            config=self.model_config,
        )

        stages = {}

        # Embed
        texts = ["test query"]
        p = self._profile("embed", lambda: self.embedder.encode(texts, normalize_embeddings=True))
        stages["embedding"] = p.to_dict()

        # Vector query
        q_emb = self.embedder.encode(texts, normalize_embeddings=True)
        p = self._profile("vector_query", lambda: self.storage.query_children(embeddings=q_emb, top_k=10))
        stages["vector_query"] = p.to_dict()

        # BM25
        bm25 = self.storage.bm25
        p = self._profile("bm25", lambda: bm25.get_scores("test query".split()))
        stages["bm25_search"] = p.to_dict()

        # RRF
        dense = [{"id": f"d{i}", "rank": i + 1} for i in range(50)]
        sparse = [{"id": f"s{i}", "rank": i + 1} for i in range(50)]
        p = self._profile("rrf", lambda: RetrievalEngine._rrf_fuse(dense, sparse))
        stages["rrf_fusion"] = p.to_dict()

        # Dedup
        items = [{"id": f"c{i}", "score": float(i), "metadata": {"parent_id": f"p{i%10}"}} for i in range(50)]
        p = self._profile("dedup", lambda: RetrievalEngine._deduplicate_by_parent(items, top_k=30))
        stages["deduplication"] = p.to_dict()

        # Rerank
        rerank_items = [{"id": f"c{i}", "text": f"content {i}"} for i in range(10)]
        p = self._profile("rerank", lambda: engine._rerank("test", rerank_items))
        stages["reranking"] = p.to_dict()

        # Budget packing
        docs = [f"doc content {i} " * 20 for i in range(20)]
        p = self._profile("packing", lambda: enforce_token_budget(docs, 5000, self.tokenizer))
        stages["budget_packing"] = p.to_dict()

        # Full pipeline
        p = self._profile("full", lambda: engine.search("Chomsky language", collect_metrics=True))
        stages["full_pipeline"] = p.to_dict()

        # Save summary
        summary_file = RESULTS_DIR / "latency_summary.json"
        summary_file.write_text(json.dumps(stages, indent=2))
        logger.info(f"Latency summary saved to {summary_file}")

        # Print formatted summary
        logger.info("=" * 70)
        logger.info("LATENCY PROFILE SUMMARY (mock models, 5 iterations)")
        logger.info("=" * 70)
        for stage, data in stages.items():
            logger.info(
                f"  {stage:20s}: mean={data['mean_ms']:>8.3f}ms  "
                f"median={data['median_ms']:>8.3f}ms  "
                f"p95={data['p95_ms']:>8.3f}ms  "
                f"std={data['std_ms']:>8.3f}ms"
            )
        logger.info("=" * 70)
