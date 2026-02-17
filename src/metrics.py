"""Metrics collection and logging for the retrieval pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetMetrics:
    """Token budget utilization metrics."""
    budget_tokens: int = 0
    used_tokens: int = 0
    utilization_pct: float = 0.0
    avg_doc_tokens: float = 0.0
    docs_packed: int = 0
    docs_skipped: int = 0
    docs_truncated: int = 0


@dataclass
class TimingMetrics:
    """Stage-level timing metrics in milliseconds."""
    query_embedding_ms: float = 0.0
    hybrid_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    dedup_ms: float = 0.0
    budget_packing_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RerankerMetrics:
    """Reranker score distribution metrics."""
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0
    items_reranked: int = 0


@dataclass
class DeduplicationMetrics:
    """Deduplication impact metrics."""
    children_before_dedup: int = 0
    children_after_dedup: int = 0
    reduction_pct: float = 0.0
    parents_deduplicated: int = 0


@dataclass
class ThresholdMetrics:
    """Reranker threshold filtering metrics."""
    threshold_value: float = 0.0
    items_before_threshold: int = 0
    items_after_threshold: int = 0
    safety_net_triggered: bool = False
    min_docs: int = 0


@dataclass
class RetrievalMetrics:
    """Aggregated metrics for a single retrieval operation."""
    budget: BudgetMetrics = field(default_factory=BudgetMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    reranker: RerankerMetrics = field(default_factory=RerankerMetrics)
    deduplication: DeduplicationMetrics = field(default_factory=DeduplicationMetrics)
    threshold: ThresholdMetrics = field(default_factory=ThresholdMetrics)
    query: str = ""
    mode: str = ""


def compute_reranker_stats(scores: list[float]) -> RerankerMetrics:
    """Compute reranker score distribution statistics."""
    if not scores:
        return RerankerMetrics()

    n = len(scores)
    score_min = min(scores)
    score_max = max(scores)
    score_mean = sum(scores) / n
    score_std = (sum((s - score_mean) ** 2 for s in scores) / (n - 1)) ** 0.5 if n > 1 else 0.0

    return RerankerMetrics(
        score_min=score_min, score_max=score_max, score_mean=score_mean,
        score_std=score_std, items_reranked=n,
    )


def log_metrics(
    metrics: RetrievalMetrics,
    mode: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Pretty-print retrieval metrics to logger."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"RETRIEVAL METRICS | Mode: {mode}")
    logger.info("=" * 60)

    b = metrics.budget
    logger.info("TOKEN BUDGET:")
    logger.info(f"   Budget: {b.budget_tokens:,} tokens")
    logger.info(f"   Used:   {b.used_tokens:,} tokens ({b.utilization_pct:.1f}%)")
    logger.info(f"   Docs packed: {b.docs_packed} | Skipped: {b.docs_skipped} | Truncated: {b.docs_truncated}")
    if b.docs_packed > 0:
        logger.info(f"   Avg doc size: {b.avg_doc_tokens:.0f} tokens")

    t = metrics.timing
    logger.info("TIMING:")
    logger.info(f"   Query embed:    {t.query_embedding_ms:>8.1f} ms")
    logger.info(f"   Hybrid search:  {t.hybrid_search_ms:>8.1f} ms")
    logger.info(f"   Sparse search:  {t.sparse_search_ms:>8.1f} ms")
    logger.info(f"   RRF fusion:     {t.rrf_fusion_ms:>8.1f} ms")
    logger.info(f"   Reranking:      {t.rerank_ms:>8.1f} ms")
    logger.info(f"   Deduplication:  {t.dedup_ms:>8.1f} ms")
    logger.info(f"   Budget packing: {t.budget_packing_ms:>8.1f} ms")
    logger.info(f"   -----------------------------")
    logger.info(f"   TOTAL:          {t.total_ms:>8.1f} ms")

    r = metrics.reranker
    if r.items_reranked > 0:
        logger.info("RERANKER SCORES:")
        logger.info(f"   Items reranked: {r.items_reranked}")
        logger.info(f"   Min: {r.score_min:.4f} | Max: {r.score_max:.4f}")
        logger.info(f"   Mean: {r.score_mean:.4f} | Std: {r.score_std:.4f}")

    d = metrics.deduplication
    if d.children_before_dedup > 0:
        logger.info("DEDUPLICATION:")
        logger.info(f"   Before: {d.children_before_dedup} children")
        logger.info(f"   After:  {d.children_after_dedup} children")
        logger.info(f"   Reduction: {d.reduction_pct:.1f}% ({d.parents_deduplicated} parents removed)")

    th = metrics.threshold
    if th.items_before_threshold > 0:
        logger.info("THRESHOLD FILTER:")
        logger.info(f"   Threshold: {th.threshold_value:.4f}")
        logger.info(f"   Before: {th.items_before_threshold} docs")
        logger.info(f"   After: {th.items_after_threshold} docs")
        if th.safety_net_triggered:
            logger.info(f"   WARNING: Safety net triggered (min: {th.min_docs})")
        else:
            reduction = ((th.items_before_threshold - th.items_after_threshold) / th.items_before_threshold) * 100
            logger.info(f"   Filtered: {reduction:.1f}% reduction")
    
    logger.info("=" * 60)


def format_metrics_summary(metrics: RetrievalMetrics) -> str:
    """Format a one-line summary of key metrics for user display."""
    b = metrics.budget
    t = metrics.timing
    
    parts = [
        f"Budget: {b.used_tokens:,}/{b.budget_tokens:,} ({b.utilization_pct:.0f}%)",
        f"Docs: {b.docs_packed}",
        f"Time: {t.total_ms:.0f}ms",
    ]
    
    if metrics.deduplication.children_before_dedup > 0:
        d = metrics.deduplication
        parts.append(f"Dedup: -{d.reduction_pct:.0f}%")
    
    return " | ".join(parts)
