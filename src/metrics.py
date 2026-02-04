"""Metrics collection and logging for the RAG retrieval pipeline.

This module provides comprehensive metrics tracking for:
- Token budget utilization
- Stage-level timing (dense search, sparse search, RRF, rerank, dedup)
- Reranker score distribution
- Deduplication impact

Usage:
    from metrics import RetrievalMetrics, log_metrics
    
    metrics = RetrievalMetrics(...)
    log_metrics(metrics, mode="regular", logger=logger)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetMetrics:
    """Token budget utilization metrics.
    
    Attributes:
        budget_tokens: Maximum allowed tokens for retrieved context
        used_tokens: Actual tokens used in final context
        utilization_pct: Percentage of budget utilized (0-100)
        avg_doc_tokens: Average tokens per packed document
        docs_packed: Number of documents included in final context
        docs_skipped: Number of documents skipped due to budget
        docs_truncated: Number of documents truncated to fit budget
    """
    budget_tokens: int = 0
    used_tokens: int = 0
    utilization_pct: float = 0.0
    avg_doc_tokens: float = 0.0
    docs_packed: int = 0
    docs_skipped: int = 0
    docs_truncated: int = 0


@dataclass
class TimingMetrics:
    """Stage-level timing metrics in milliseconds.
    
    Attributes:
        dense_search_ms: Time for dense (embedding) search
        sparse_search_ms: Time for sparse (BM25) search
        rrf_fusion_ms: Time for RRF fusion
        rerank_ms: Time for reranking
        dedup_ms: Time for deduplication
        budget_packing_ms: Time for token budget packing
        total_ms: Total retrieval pipeline time
    """
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    dedup_ms: float = 0.0
    budget_packing_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RerankerMetrics:
    """Reranker score distribution metrics.
    
    Attributes:
        score_min: Minimum reranker score
        score_max: Maximum reranker score
        score_mean: Mean reranker score
        score_std: Standard deviation of reranker scores
        items_reranked: Number of items passed through reranker
    """
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0
    items_reranked: int = 0


@dataclass
class DeduplicationMetrics:
    """Deduplication impact metrics.
    
    Attributes:
        children_before_dedup: Number of children before deduplication
        children_after_dedup: Number of children after deduplication
        reduction_pct: Percentage reduction from deduplication
        parents_deduplicated: Number of duplicate parents removed
    """
    children_before_dedup: int = 0
    children_after_dedup: int = 0
    reduction_pct: float = 0.0
    parents_deduplicated: int = 0


@dataclass
class ThresholdMetrics:
    """Reranker threshold filtering metrics.
    
    Attributes:
        threshold_value: The reranker score threshold used
        items_before_threshold: Number of items before threshold filtering
        items_after_threshold: Number of items after threshold filtering
        safety_net_triggered: Whether minimum document safety net was used
        min_docs: Minimum documents setting
    """
    threshold_value: float = 0.0
    items_before_threshold: int = 0
    items_after_threshold: int = 0
    safety_net_triggered: bool = False
    min_docs: int = 0


@dataclass
class RetrievalMetrics:
    """Comprehensive metrics for a single retrieval operation.
    
    Aggregates budget, timing, reranker, deduplication, and threshold metrics
    for analysis and logging.
    
    Attributes:
        budget: Token budget utilization metrics
        timing: Stage-level timing metrics
        reranker: Reranker score distribution metrics
        deduplication: Deduplication impact metrics
        threshold: Reranker threshold filtering metrics
        query: The original query (for correlation)
        mode: The RAG mode used (regular, power-fast, power-deep-research)
    """
    budget: BudgetMetrics = field(default_factory=BudgetMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    reranker: RerankerMetrics = field(default_factory=RerankerMetrics)
    deduplication: DeduplicationMetrics = field(default_factory=DeduplicationMetrics)
    threshold: ThresholdMetrics = field(default_factory=ThresholdMetrics)
    query: str = ""
    mode: str = ""


def compute_reranker_stats(scores: list[float]) -> RerankerMetrics:
    """Compute reranker score distribution statistics.
    
    Args:
        scores: List of reranker scores
        
    Returns:
        RerankerMetrics with computed statistics
    """
    if not scores:
        return RerankerMetrics()
    
    n = len(scores)
    score_min = min(scores)
    score_max = max(scores)
    score_mean = sum(scores) / n
    
    # Compute standard deviation
    if n > 1:
        variance = sum((s - score_mean) ** 2 for s in scores) / (n - 1)
        score_std = variance ** 0.5
    else:
        score_std = 0.0
    
    return RerankerMetrics(
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        items_reranked=n,
    )


def log_metrics(
    metrics: RetrievalMetrics,
    mode: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Pretty-print retrieval metrics to logger.
    
    Formats all metrics in a human-readable format suitable for
    debugging and performance analysis.
    
    Args:
        metrics: RetrievalMetrics instance to log
        mode: Current RAG mode for context
        logger: Logger instance (uses module logger if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Header
    logger.info("=" * 60)
    logger.info(f"RETRIEVAL METRICS | Mode: {mode}")
    logger.info("=" * 60)
    
    # Budget metrics
    b = metrics.budget
    logger.info("TOKEN BUDGET:")
    logger.info(f"   Budget: {b.budget_tokens:,} tokens")
    logger.info(f"   Used:   {b.used_tokens:,} tokens ({b.utilization_pct:.1f}%)")
    logger.info(f"   Docs packed: {b.docs_packed} | Skipped: {b.docs_skipped} | Truncated: {b.docs_truncated}")
    if b.docs_packed > 0:
        logger.info(f"   Avg doc size: {b.avg_doc_tokens:.0f} tokens")
    
    # Timing metrics
    t = metrics.timing
    logger.info("TIMING:")
    logger.info(f"   Dense search:   {t.dense_search_ms:>8.1f} ms")
    logger.info(f"   Sparse search:  {t.sparse_search_ms:>8.1f} ms")
    logger.info(f"   RRF fusion:     {t.rrf_fusion_ms:>8.1f} ms")
    logger.info(f"   Reranking:      {t.rerank_ms:>8.1f} ms")
    logger.info(f"   Deduplication:  {t.dedup_ms:>8.1f} ms")
    logger.info(f"   Budget packing: {t.budget_packing_ms:>8.1f} ms")
    logger.info(f"   -----------------------------")
    logger.info(f"   TOTAL:          {t.total_ms:>8.1f} ms")
    
    # Reranker metrics
    r = metrics.reranker
    if r.items_reranked > 0:
        logger.info("RERANKER SCORES:")
        logger.info(f"   Items reranked: {r.items_reranked}")
        logger.info(f"   Min: {r.score_min:.4f} | Max: {r.score_max:.4f}")
        logger.info(f"   Mean: {r.score_mean:.4f} | Std: {r.score_std:.4f}")
    
    # Deduplication metrics
    d = metrics.deduplication
    if d.children_before_dedup > 0:
        logger.info("DEDUPLICATION:")
        logger.info(f"   Before: {d.children_before_dedup} children")
        logger.info(f"   After:  {d.children_after_dedup} children")
        logger.info(f"   Reduction: {d.reduction_pct:.1f}% ({d.parents_deduplicated} parents removed)")
    
    # Threshold filtering metrics
    th = metrics.threshold
    if th.items_before_threshold > 0:
        logger.info("THRESHOLD FILTER:")
        logger.info(f"   Threshold: {th.threshold_value:.1f}")
        logger.info(f"   Before: {th.items_before_threshold} docs")
        logger.info(f"   After: {th.items_after_threshold} docs")
        if th.safety_net_triggered:
            logger.info(f"   WARNING: Safety net triggered (min: {th.min_docs})")
        else:
            reduction = ((th.items_before_threshold - th.items_after_threshold) / th.items_before_threshold) * 100
            logger.info(f"   Filtered: {reduction:.1f}% reduction")
    
    logger.info("=" * 60)


def format_metrics_summary(metrics: RetrievalMetrics) -> str:
    """Format a one-line summary of key metrics for user display.
    
    Args:
        metrics: RetrievalMetrics instance
        
    Returns:
        Formatted summary string
    """
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
