"""Optimisation experiments for the RAG pipeline.

Five controlled experiments measuring the impact of specific changes:
1. Vector index warmup
2. Deduplication ordering (before vs after reranking)
3. Real model profiling (embedding + reranker on Apple Silicon)
4. Query embedding caching
5. Token budget utilisation with realistic document sizes

All experiments reuse the same fixed corpus and queries from conftest.py
and log results in structured JSON for the consolidated report.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from tests.hardware_info import HARDWARE

from src.config import ModelConfig, _get_mode_config
from src.generator import enforce_token_budget, count_tokens, BudgetPackResult
from src.retrieval import RetrievalEngine, RetrievalResult
from src.storage import StorageConfig, StorageEngine

logger = get_test_logger("experiments")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Queries used across experiments (skip empty string edge case)
EXPERIMENT_QUERIES = [q for q in FIXED_QUERIES if q.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_storage(tmp_path: Path, embedder: MockEmbeddingModel) -> StorageEngine:
    """Create a fresh StorageEngine with the fixed corpus loaded."""
    config = StorageConfig(
        lance_dir=tmp_path / "lance",
        lance_table="test_chunks",
    )
    engine = StorageEngine(config)
    parents, children = generate_parent_child_corpus()
    engine.add_parents(parents)
    texts = [c.text for c in children]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    engine.add_children(children, embeddings=embeddings)
    return engine


def _build_retrieval_engine(
    storage: StorageEngine,
    embedder: MockEmbeddingModel,
    reranker: MockReranker,
    mode: str = "regular",
    ram_gb: float = 32.0,
) -> RetrievalEngine:
    config = _get_mode_config(mode, ram_gb)
    return RetrievalEngine(
        storage=storage,
        embedding_model=embedder,
        reranker=reranker,
        config=config,
    )


def _profile_fn(fn, iterations: int = 10) -> dict[str, float]:
    """Run fn() multiple times and return timing statistics in ms."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.mean(times), 4),
        "median_ms": round(statistics.median(times), 4),
        "p95_ms": round(sorted(times)[int(0.95 * len(times))], 4),
        "std_ms": round(statistics.stdev(times) if len(times) > 1 else 0, 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "iterations": iterations,
    }


# ===================================================================
# EXPERIMENT 1: Vector Index Warmup
# ===================================================================

class TestExp1VectorWarmup:
    """Measure the effect of a warmup query on cold-start latency."""

    def test_cold_vs_warm_latency(self, tmp_path: Path):
        """Compare first-query latency with and without warmup."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        results = {"experiment": "vector_index_warmup", "hardware": HARDWARE.as_dict()}

        # --- Cold start (no warmup) ---
        cold_latencies = []
        for trial in range(5):
            storage = _create_storage(tmp_path / f"cold_{trial}", embedder)
            engine = _build_retrieval_engine(storage, embedder, reranker)
            t0 = time.perf_counter()
            engine.search("test query for cold start")
            cold_latencies.append((time.perf_counter() - t0) * 1000)
            storage.close()

        # --- Warm start (with warmup query) ---
        warm_first_latencies = []
        warm_second_latencies = []
        for trial in range(5):
            storage = _create_storage(tmp_path / f"warm_{trial}", embedder)
            engine = _build_retrieval_engine(storage, embedder, reranker)
            # Warmup: run a throwaway query
            t0 = time.perf_counter()
            engine.search("warmup query")
            warmup_ms = (time.perf_counter() - t0) * 1000
            warm_first_latencies.append(warmup_ms)
            # Actual query
            t0 = time.perf_counter()
            engine.search("test query for warm start")
            warm_second_latencies.append((time.perf_counter() - t0) * 1000)
            storage.close()

        results["cold_start"] = {
            "mean_ms": round(statistics.mean(cold_latencies), 4),
            "median_ms": round(statistics.median(cold_latencies), 4),
            "p95_ms": round(sorted(cold_latencies)[int(0.95 * len(cold_latencies))], 4),
            "std_ms": round(statistics.stdev(cold_latencies), 4),
            "all_ms": [round(x, 4) for x in cold_latencies],
        }
        results["warmup_query"] = {
            "mean_ms": round(statistics.mean(warm_first_latencies), 4),
            "median_ms": round(statistics.median(warm_first_latencies), 4),
        }
        results["after_warmup"] = {
            "mean_ms": round(statistics.mean(warm_second_latencies), 4),
            "median_ms": round(statistics.median(warm_second_latencies), 4),
            "p95_ms": round(sorted(warm_second_latencies)[int(0.95 * len(warm_second_latencies))], 4),
            "std_ms": round(statistics.stdev(warm_second_latencies), 4),
            "all_ms": [round(x, 4) for x in warm_second_latencies],
        }

        cold_mean = statistics.mean(cold_latencies)
        warm_mean = statistics.mean(warm_second_latencies)
        improvement_pct = ((cold_mean - warm_mean) / cold_mean * 100) if cold_mean > 0 else 0
        results["improvement_pct"] = round(improvement_pct, 2)
        results["conclusion"] = (
            f"Warmup reduces mean first-query latency by {improvement_pct:.1f}% "
            f"(cold={cold_mean:.3f}ms -> warm={warm_mean:.3f}ms)"
        )

        logger.info(results["conclusion"])
        _save_result("exp1_warmup.json", results)

        # Warmup should not make things worse
        assert warm_mean <= cold_mean * 1.5, "Warmup should not significantly increase latency"

    def test_warmup_multi_query_stability(self, tmp_path: Path):
        """Verify that post-warmup latencies are stable across queries."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        storage = _create_storage(tmp_path / "stability", embedder)
        engine = _build_retrieval_engine(storage, embedder, reranker)

        # Warmup
        engine.search("warmup query")

        # Measure all queries
        latencies = {}
        for q in EXPERIMENT_QUERIES:
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                engine.search(q)
                times.append((time.perf_counter() - t0) * 1000)
            latencies[q] = {
                "mean_ms": round(statistics.mean(times), 4),
                "std_ms": round(statistics.stdev(times), 4),
            }

        storage.close()

        # Check variance is low (coefficient of variation < 100%)
        all_means = [v["mean_ms"] for v in latencies.values()]
        cv = statistics.stdev(all_means) / statistics.mean(all_means) if len(all_means) > 1 else 0
        assert cv < 1.0, f"Post-warmup queries have high variance: CV={cv:.2f}"


# ===================================================================
# EXPERIMENT 2: Deduplication Order
# ===================================================================

class TestExp2DedupOrder:
    """Compare dedup-before-rerank vs dedup-after-rerank."""

    def test_dedup_before_vs_after_rerank(self, tmp_path: Path):
        """Measure latency and result quality for both orderings."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        storage = _create_storage(tmp_path / "dedup_order", embedder)
        config = _get_mode_config("regular", 32.0)
        results = {"experiment": "dedup_order", "hardware": HARDWARE.as_dict()}

        query_results = []
        for q in EXPERIMENT_QUERIES:
            # --- Current order: hybrid → dedup → rerank (as in retrieval.py) ---
            engine_current = RetrievalEngine(
                storage=storage, embedding_model=embedder,
                reranker=reranker, config=config,
            )
            t0 = time.perf_counter()
            current_results = engine_current.search(q)
            current_ms = (time.perf_counter() - t0) * 1000

            # --- Alternative: hybrid → rerank → dedup ---
            t0 = time.perf_counter()
            fused = engine_current._hybrid_search(q, config.top_k_fused)

            # Hydrate missing text/metadata
            missing_ids = [item["id"] for item in fused if "text" not in item or "metadata" not in item]
            if missing_ids:
                fetched = storage.get_children_by_ids(missing_ids)
                for item in fused:
                    if item["id"] in fetched:
                        item.setdefault("text", fetched[item["id"]].get("text"))
                        item.setdefault("metadata", fetched[item["id"]].get("metadata"))

            # Expand parent text for reranking
            parent_cache: dict[str, str] = {}
            for item in fused:
                metadata = item.get("metadata") or {}
                parent_id = metadata.get("parent_id")
                if isinstance(parent_id, str) and parent_id not in parent_cache:
                    parent_text = storage.get_parent_text(parent_id)
                    if parent_text:
                        parent_cache[parent_id] = parent_text
                if isinstance(parent_id, str) and parent_id in parent_cache:
                    item["rerank_text"] = parent_cache[parent_id]

            # Rerank FIRST (no dedup yet)
            reranked, raw_scores = engine_current._rerank(q, fused[:config.top_k_rerank])
            reranker_input_size = len(fused[:config.top_k_rerank])

            # THEN dedup
            deduped, dedup_metrics = engine_current._deduplicate_by_parent(reranked, config.top_k_final)
            alt_ms = (time.perf_counter() - t0) * 1000

            query_results.append({
                "query": q,
                "current_order": {
                    "label": "dedup_before_rerank",
                    "latency_ms": round(current_ms, 4),
                    "result_count": len(current_results),
                    "reranker_input_size": "N/A (internal)",
                },
                "alternative_order": {
                    "label": "dedup_after_rerank",
                    "latency_ms": round(alt_ms, 4),
                    "result_count": len(deduped),
                    "reranker_input_size": reranker_input_size,
                },
            })

        storage.close()

        # Aggregate
        current_latencies = [r["current_order"]["latency_ms"] for r in query_results]
        alt_latencies = [r["alternative_order"]["latency_ms"] for r in query_results]
        results["per_query"] = query_results
        results["summary"] = {
            "current_mean_ms": round(statistics.mean(current_latencies), 4),
            "alternative_mean_ms": round(statistics.mean(alt_latencies), 4),
            "current_median_ms": round(statistics.median(current_latencies), 4),
            "alternative_median_ms": round(statistics.median(alt_latencies), 4),
        }

        diff_pct = (
            (statistics.mean(alt_latencies) - statistics.mean(current_latencies))
            / statistics.mean(current_latencies) * 100
        ) if statistics.mean(current_latencies) > 0 else 0
        results["summary"]["latency_diff_pct"] = round(diff_pct, 2)
        results["conclusion"] = (
            f"Dedup-after-rerank is {abs(diff_pct):.1f}% "
            f"{'slower' if diff_pct > 0 else 'faster'} than current order. "
            f"Current order deduplicates before reranking, reducing reranker input size."
        )

        logger.info(results["conclusion"])
        _save_result("exp2_dedup_order.json", results)

    def test_dedup_order_reranker_batch_impact(self, tmp_path: Path):
        """Measure how dedup ordering affects reranker batch size."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        storage = _create_storage(tmp_path / "dedup_batch", embedder)
        config = _get_mode_config("regular", 32.0)
        engine = RetrievalEngine(
            storage=storage, embedding_model=embedder,
            reranker=reranker, config=config,
        )

        for q in EXPERIMENT_QUERIES:
            # Current pipeline: hybrid → dedup → rerank
            fused = engine._hybrid_search(q, config.top_k_fused)

            before_dedup = len(fused)
            deduped, _ = engine._deduplicate_by_parent(fused, config.top_k_fused)
            after_dedup = len(deduped)

            reranker.call_count = 0
            engine._rerank(q, deduped[:config.top_k_rerank])
            batch_size_current = reranker.last_batch_size

            reranker.call_count = 0
            engine._rerank(q, fused[:config.top_k_rerank])
            batch_size_alternative = reranker.last_batch_size

            logger.info(
                f"Q: {q[:40]} | before_dedup={before_dedup} after_dedup={after_dedup} "
                f"rerank_batch_current={batch_size_current} rerank_batch_alt={batch_size_alternative}"
            )

            # Dedup-first should produce smaller or equal reranker batch
            assert batch_size_current <= batch_size_alternative

        storage.close()


# ===================================================================
# EXPERIMENT 3: Real Model Profiling
# ===================================================================

class TestExp3RealModelProfiling:
    """Profile real embedding model and reranker on Apple Silicon.

    These tests are conditional: they skip if models are not available.
    When models ARE available, they measure actual inference latency
    and memory usage.
    """

    @pytest.fixture
    def real_embedder(self):
        """Try to load the real BAAI/bge-m3 embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-m3")
            return model
        except Exception:
            pytest.skip("BAAI/bge-m3 not available (not downloaded)")

    @pytest.fixture
    def real_reranker(self):
        """Try to load the real Jina reranker."""
        try:
            from src.reranker import JinaRerankerMLX
            reranker = JinaRerankerMLX()
            return reranker
        except Exception:
            pytest.skip("Jina reranker not available (not downloaded)")

    def test_embedding_encode_latency(self, real_embedder):
        """Measure real embedding encode time."""
        results = {"stage": "embedding_encode", "hardware": HARDWARE.as_dict()}
        texts = [item["text"] for item in FIXED_CORPUS]

        # Warmup
        real_embedder.encode(["warmup text"], normalize_embeddings=True)

        # Single text encoding
        single_profile = _profile_fn(
            lambda: real_embedder.encode([texts[0]], normalize_embeddings=True),
            iterations=10,
        )
        results["single_text"] = single_profile

        # Batch encoding (all corpus docs)
        batch_profile = _profile_fn(
            lambda: real_embedder.encode(texts, normalize_embeddings=True),
            iterations=10,
        )
        results["batch_5_texts"] = batch_profile

        # Per-text in batch
        results["per_text_in_batch_ms"] = round(
            batch_profile["mean_ms"] / len(texts), 4
        )

        logger.info(
            f"Embedding encode: single={single_profile['mean_ms']:.2f}ms, "
            f"batch(5)={batch_profile['mean_ms']:.2f}ms, "
            f"per_text_in_batch={results['per_text_in_batch_ms']:.2f}ms"
        )
        _save_result("exp3_embedding_latency.json", results)

    def test_reranker_latency_by_batch_size(self, real_reranker):
        """Measure reranker latency as a function of batch size."""
        results = {"stage": "reranker_batch_scaling", "hardware": HARDWARE.as_dict()}
        query = "What is Chomsky's theory of language?"
        docs = [item["text"] for item in FIXED_CORPUS]

        batch_profiles = {}
        for batch_size in [1, 2, 3, 5]:
            pairs = [(query, doc) for doc in docs[:batch_size]]
            # Warmup
            real_reranker.compute_score(pairs)
            profile = _profile_fn(
                lambda p=pairs: real_reranker.compute_score(p),
                iterations=5,
            )
            batch_profiles[str(batch_size)] = profile
            logger.info(f"Reranker batch={batch_size}: mean={profile['mean_ms']:.2f}ms")

        results["batch_profiles"] = batch_profiles
        results["scaling_factor"] = round(
            batch_profiles["5"]["mean_ms"] / batch_profiles["1"]["mean_ms"], 2
        ) if batch_profiles.get("1", {}).get("mean_ms", 0) > 0 else None

        _save_result("exp3_reranker_batch_scaling.json", results)

    def test_real_pipeline_latency(self, tmp_path: Path, real_embedder):
        """Measure full pipeline with real embeddings (mock reranker)."""
        results = {"stage": "real_pipeline", "hardware": HARDWARE.as_dict()}

        # Build storage with real embeddings
        config = StorageConfig(
            lance_dir=tmp_path / "real_lance",
            lance_table="real_chunks",
        )
        storage = StorageEngine(config)
        parents, children = generate_parent_child_corpus()
        storage.add_parents(parents)
        texts = [c.text for c in children]

        t0 = time.perf_counter()
        embeddings = real_embedder.encode(texts, normalize_embeddings=True).tolist()
        ingest_embed_ms = (time.perf_counter() - t0) * 1000
        results["ingest_embedding_ms"] = round(ingest_embed_ms, 2)

        storage.add_children(children, embeddings=embeddings)

        mock_reranker = MockReranker()
        pipeline_config = _get_mode_config("regular", 32.0)
        engine = RetrievalEngine(
            storage=storage, embedding_model=real_embedder,
            reranker=mock_reranker, config=pipeline_config,
        )

        # Warmup
        engine.search("warmup query")

        query_latencies = {}
        for q in EXPERIMENT_QUERIES:
            profile = _profile_fn(lambda qq=q: engine.search(qq), iterations=5)
            query_latencies[q] = profile

        results["query_latencies"] = query_latencies
        all_means = [v["mean_ms"] for v in query_latencies.values()]
        results["overall_mean_ms"] = round(statistics.mean(all_means), 4)
        results["overall_p95_ms"] = round(
            sorted(all_means)[int(0.95 * len(all_means))], 4
        )

        storage.close()
        logger.info(
            f"Real pipeline: mean={results['overall_mean_ms']:.2f}ms, "
            f"ingest_embed={ingest_embed_ms:.2f}ms"
        )
        _save_result("exp3_real_pipeline.json", results)

    def test_memory_usage_profiling(self, real_embedder):
        """Measure memory usage for embedding operations."""
        results = {"stage": "memory_profiling", "hardware": HARDWARE.as_dict()}
        texts = [item["text"] for item in FIXED_CORPUS]

        gc.collect()
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        _ = real_embedder.encode(texts, normalize_embeddings=True)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_diff_kb = sum(s.size_diff for s in stats) / 1024
        results["embedding_memory_delta_kb"] = round(total_diff_kb, 2)
        results["texts_encoded"] = len(texts)
        results["per_text_kb"] = round(total_diff_kb / len(texts), 2) if texts else 0

        logger.info(f"Memory: {total_diff_kb:.1f} KB for {len(texts)} texts ({results['per_text_kb']:.1f} KB/text)")
        _save_result("exp3_memory.json", results)


# ===================================================================
# EXPERIMENT 4: Query Embedding Caching
# ===================================================================

class TestExp4Caching:
    """Measure the impact of caching query embeddings."""

    def test_cache_vs_no_cache(self, tmp_path: Path):
        """Compare repeated query latency with and without embedding cache."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        storage = _create_storage(tmp_path / "cache_test", embedder)
        config = _get_mode_config("regular", 32.0)

        results = {"experiment": "embedding_cache", "hardware": HARDWARE.as_dict()}

        # --- No cache: raw encoder ---
        engine_no_cache = RetrievalEngine(
            storage=storage, embedding_model=embedder,
            reranker=reranker, config=config,
        )
        # Warmup
        engine_no_cache.search("warmup")

        no_cache_latencies = {}
        for q in EXPERIMENT_QUERIES:
            times = []
            for _ in range(10):
                embedder.call_count = 0
                t0 = time.perf_counter()
                engine_no_cache.search(q)
                times.append((time.perf_counter() - t0) * 1000)
            no_cache_latencies[q] = {
                "mean_ms": round(statistics.mean(times), 4),
                "median_ms": round(statistics.median(times), 4),
                "encode_calls": 10,  # 1 per search
            }

        # --- With cache: wrap encoder ---
        cached_embedder = _CachingEmbedder(embedder)
        engine_cached = RetrievalEngine(
            storage=storage, embedding_model=cached_embedder,
            reranker=reranker, config=config,
        )
        engine_cached.search("warmup")
        cached_embedder.cache.clear()  # clear warmup from cache

        cached_latencies = {}
        for q in EXPERIMENT_QUERIES:
            times = []
            for rep in range(10):
                cached_embedder.call_count = 0
                t0 = time.perf_counter()
                engine_cached.search(q)
                times.append((time.perf_counter() - t0) * 1000)
            cached_latencies[q] = {
                "mean_ms": round(statistics.mean(times), 4),
                "median_ms": round(statistics.median(times), 4),
                "cache_hits": cached_embedder.cache_hits,
                "cache_misses": cached_embedder.cache_misses,
            }

        storage.close()

        results["no_cache"] = no_cache_latencies
        results["with_cache"] = cached_latencies

        # Overall comparison
        no_cache_means = [v["mean_ms"] for v in no_cache_latencies.values()]
        cached_means = [v["mean_ms"] for v in cached_latencies.values()]
        nc_avg = statistics.mean(no_cache_means)
        c_avg = statistics.mean(cached_means)
        improvement = ((nc_avg - c_avg) / nc_avg * 100) if nc_avg > 0 else 0

        results["summary"] = {
            "no_cache_mean_ms": round(nc_avg, 4),
            "cached_mean_ms": round(c_avg, 4),
            "improvement_pct": round(improvement, 2),
            "total_cache_hits": cached_embedder.cache_hits,
            "total_cache_misses": cached_embedder.cache_misses,
            "cache_hit_rate_pct": round(
                cached_embedder.cache_hits
                / max(cached_embedder.cache_hits + cached_embedder.cache_misses, 1)
                * 100, 1
            ),
        }
        results["conclusion"] = (
            f"Caching improves repeated-query latency by {improvement:.1f}%. "
            f"Cache hit rate: {results['summary']['cache_hit_rate_pct']:.0f}%. "
            f"Benefit scales with embedding model cost (mock embedder is fast; "
            f"real models will show larger gains)."
        )

        logger.info(results["conclusion"])
        _save_result("exp4_caching.json", results)

    def test_cache_correctness(self, tmp_path: Path):
        """Verify that cached results are identical to non-cached."""
        embedder = MockEmbeddingModel(dim=384)
        reranker = MockReranker()
        storage = _create_storage(tmp_path / "cache_correct", embedder)
        config = _get_mode_config("regular", 32.0)

        engine_direct = RetrievalEngine(
            storage=storage, embedding_model=embedder,
            reranker=reranker, config=config,
        )
        cached_embedder = _CachingEmbedder(embedder)
        engine_cached = RetrievalEngine(
            storage=storage, embedding_model=cached_embedder,
            reranker=reranker, config=config,
        )

        for q in EXPERIMENT_QUERIES:
            direct = engine_direct.search(q)
            cached = engine_cached.search(q)
            assert len(direct) == len(cached), f"Result count mismatch for '{q}'"
            for d, c in zip(direct, cached):
                assert d.child_id == c.child_id, f"Child ID mismatch for '{q}'"
                assert d.text == c.text, f"Text mismatch for '{q}'"

        storage.close()


class _CachingEmbedder:
    """Wrapper around an embedding model that caches query embeddings."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.cache: dict[str, list[list[float]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.call_count = 0

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        self.call_count += 1
        key = "|".join(texts)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        result = self._inner.encode(texts, **kwargs)
        self.cache[key] = result
        return result


# ===================================================================
# EXPERIMENT 5: Token Budget Utilisation
# ===================================================================

# Realistic longer documents for budget experiments
REALISTIC_DOCS = [
    # ~300 words each to test budget packing more realistically
    " ".join(
        f"This is sentence {i} of document {d} discussing topic {chr(65+d)}. "
        f"The analysis covers aspects of {'linguistics' if d % 2 == 0 else 'philosophy'} "
        f"in depth with multiple clauses and technical terminology. "
        for i in range(50)
    )
    for d in range(15)
]


class TestExp5BudgetUtilisation:
    """Analyse token budget utilisation with realistic document sizes."""

    def test_budget_utilisation_by_mode(self):
        """Measure how much of the context budget is actually used per mode."""
        tokenizer = MockTokenizer()
        results = {"experiment": "budget_utilisation", "hardware": HARDWARE.as_dict()}

        mode_results = {}
        for mode in ["regular", "power-deep-research"]:
            config = _get_mode_config(mode, 32.0)
            budget = config.retrieval_budget

            # Pack realistic docs
            pack_result = enforce_token_budget(
                REALISTIC_DOCS,
                budget,
                tokenizer,
                allow_truncation=True,
            )

            utilisation = (pack_result.used_tokens / budget * 100) if budget > 0 else 0
            mode_results[mode] = {
                "budget_tokens": budget,
                "used_tokens": pack_result.used_tokens,
                "utilisation_pct": round(utilisation, 2),
                "packed_docs": len(pack_result.packed_docs),
                "total_docs_available": len(REALISTIC_DOCS),
                "skipped_count": pack_result.skipped_count,
                "truncated_count": pack_result.truncated_count,
                "consecutive_fails": pack_result.consecutive_fails,
            }

        results["by_mode"] = mode_results

        # Also test with varied budget sizes
        budget_sweep = {}
        for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
            pack_result = enforce_token_budget(
                REALISTIC_DOCS, budget, tokenizer, allow_truncation=True,
            )
            utilisation = (pack_result.used_tokens / budget * 100) if budget > 0 else 0
            budget_sweep[str(budget)] = {
                "used_tokens": pack_result.used_tokens,
                "utilisation_pct": round(utilisation, 2),
                "packed_docs": len(pack_result.packed_docs),
                "skipped_count": pack_result.skipped_count,
                "truncated_count": pack_result.truncated_count,
            }

        results["budget_sweep"] = budget_sweep
        results["conclusion"] = (
            f"Budget utilisation ranges from "
            f"{min(v['utilisation_pct'] for v in mode_results.values()):.0f}% "
            f"to {max(v['utilisation_pct'] for v in mode_results.values()):.0f}% "
            f"across modes with {len(REALISTIC_DOCS)} realistic documents."
        )

        logger.info(results["conclusion"])
        _save_result("exp5_budget_utilisation.json", results)

    def test_truncation_behaviour(self):
        """Measure truncation frequency under tight budgets."""
        tokenizer = MockTokenizer()
        results = {"experiment": "truncation_analysis", "hardware": HARDWARE.as_dict()}

        analyses = {}
        for budget in [500, 1000, 2000, 5000]:
            # With truncation
            with_trunc = enforce_token_budget(
                REALISTIC_DOCS, budget, tokenizer, allow_truncation=True,
            )
            # Without truncation
            without_trunc = enforce_token_budget(
                REALISTIC_DOCS, budget, tokenizer, allow_truncation=False,
            )

            analyses[str(budget)] = {
                "with_truncation": {
                    "packed_docs": len(with_trunc.packed_docs),
                    "used_tokens": with_trunc.used_tokens,
                    "truncated_count": with_trunc.truncated_count,
                    "utilisation_pct": round(with_trunc.used_tokens / budget * 100, 2),
                },
                "without_truncation": {
                    "packed_docs": len(without_trunc.packed_docs),
                    "used_tokens": without_trunc.used_tokens,
                    "truncated_count": 0,
                    "utilisation_pct": round(without_trunc.used_tokens / budget * 100, 2),
                },
                "truncation_benefit_docs": (
                    len(with_trunc.packed_docs) - len(without_trunc.packed_docs)
                ),
            }

        results["analyses"] = analyses
        _save_result("exp5_truncation.json", results)

    def test_document_size_distribution(self):
        """Profile document token sizes in the realistic corpus."""
        tokenizer = MockTokenizer()
        doc_sizes = [count_tokens(doc, tokenizer) for doc in REALISTIC_DOCS]

        results = {
            "experiment": "doc_size_distribution",
            "hardware": HARDWARE.as_dict(),
            "doc_count": len(REALISTIC_DOCS),
            "sizes": doc_sizes,
            "mean_tokens": round(statistics.mean(doc_sizes), 1),
            "median_tokens": round(statistics.median(doc_sizes), 1),
            "min_tokens": min(doc_sizes),
            "max_tokens": max(doc_sizes),
            "std_tokens": round(statistics.stdev(doc_sizes), 1) if len(doc_sizes) > 1 else 0,
            "total_tokens": sum(doc_sizes),
        }

        _save_result("exp5_doc_sizes.json", results)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_result(filename: str, data: dict) -> None:
    """Save experiment result to tests/results/."""
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved: {path.name}")
