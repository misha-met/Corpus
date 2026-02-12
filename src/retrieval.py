from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .config import ModelConfig
from .metrics import (
    RetrievalMetrics,
    TimingMetrics,
    RerankerMetrics,
    DeduplicationMetrics,
    ThresholdMetrics,
    compute_reranker_stats,
)
from .storage import StorageEngine

logger = logging.getLogger(__name__)

_BOILERPLATE_PATTERNS = (
    "as an ai",
    "as a language model",
    "i do not have moral beliefs",
    "i cannot be considered",
)


def format_chunk_for_citation(
    text: str,
    source_id: str,
    display_page: Optional[str] = None,
    chunk_index: int = 1,
) -> str:
    """Format a chunk with citation markers for Academic Mode.

    When a page number is available the marker includes ``PAGE: X``.
    When page metadata is missing the PAGE field is omitted so the LLM
    cites as ``[SourceID]`` instead of ``[SourceID, p. Unknown]``.
    """
    if display_page:
        header = f"[CHUNK START | SOURCE: {source_id} | PAGE: {display_page}]"
    else:
        header = f"[CHUNK START | SOURCE: {source_id}]"
    return f"{header}\n{text.strip()}\n[CHUNK END]"


def format_context_with_citations(
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Format chunks with citation markers and build source mapping."""
    if len(texts) != len(metadatas):
        raise ValueError(f"texts and metadatas must have same length, got {len(texts)} vs {len(metadatas)}")

    formatted_chunks: list[str] = []
    source_mapping: dict[str, str] = {}
    
    for idx, (text, meta) in enumerate(zip(texts, metadatas), start=1):
        source_id = meta.get("source_id", "Unknown")
        display_page = meta.get("display_page")

        doc_name = meta.get("doc_name") or meta.get("filename")
        if doc_name and doc_name != source_id and source_id not in source_mapping:
            source_mapping[source_id] = doc_name

        formatted_chunks.append(format_chunk_for_citation(
            text=text, source_id=source_id, display_page=display_page, chunk_index=idx,
        ))
    
    context = "\n\n".join(formatted_chunks)
    return context, source_mapping


def build_source_legend(source_mapping: dict[str, str]) -> str:
    """Build a source legend for citation reference."""
    useful_mappings = {sid: name for sid, name in source_mapping.items() if sid != name}
    if not useful_mappings:
        return ""
    lines = ["SOURCE LEGEND:"]
    for source_id, doc_name in sorted(useful_mappings.items()):
        lines.append(f"- {source_id} → {doc_name}")
    return "\n".join(lines)

@dataclass(frozen=True)
class RetrievalResult:
    """Result from retrieval pipeline."""
    child_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    parent_text: Optional[str]
    metrics: Optional[RetrievalMetrics] = None


class RetrievalEngine:
    """Hybrid retrieval engine using LanceDB native hybrid search + reranking."""

    def __init__(
        self,
        *,
        storage: StorageEngine,
        embedding_model: Any,
        reranker: Optional[Any] = None,
        config: Optional[ModelConfig] = None,
    ) -> None:
        self._storage = storage
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._config = config

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        *,
        source_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Single-call hybrid search via LanceDB (vector ANN + FTS BM25 + RRF)."""
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        try:
            embeddings = self._embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
            query_vector = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else list(embeddings[0])
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("Embedding model encode failed.") from exc

        return self._storage.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            source_id=source_id,
        )

    @staticmethod
    def _deduplicate_by_parent(
        items: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], DeduplicationMetrics]:
        """Keep highest-scored child per unique parent_id (runs before reranking)."""
        before_count = len(items)
        seen_parents: dict[str, dict[str, Any]] = {}
        no_parent_items: list[dict[str, Any]] = []

        for item in items:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")

            if not isinstance(parent_id, str):
                no_parent_items.append(item)
                continue

            if parent_id not in seen_parents:
                seen_parents[parent_id] = item
            elif item.get("score", 0) > seen_parents[parent_id].get("score", 0):
                seen_parents[parent_id] = item

        deduplicated = list(seen_parents.values()) + no_parent_items
        deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)
        deduplicated_count = len(deduplicated)
        dedup_removed_count = max(0, before_count - deduplicated_count)
        top_k_limited = deduplicated[:top_k]

        metrics = DeduplicationMetrics(
            children_before_dedup=before_count,
            children_after_dedup=deduplicated_count,
            reduction_pct=100 * (1 - deduplicated_count / before_count) if before_count > 0 else 0,
            parents_deduplicated=dedup_removed_count,
        )
        return top_k_limited, metrics

    def _rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Rerank items and return both reranked items and raw scores."""
        if self._reranker is None or not items:
            return items, []

        pairs = [(query, item.get("rerank_text", item.get("text", ""))) for item in items]

        try:
            if hasattr(self._reranker, "compute_score"):
                scores = self._reranker.compute_score(pairs)
            elif hasattr(self._reranker, "predict"):
                scores = self._reranker.predict(pairs)
            else:
                raise AttributeError("Reranker missing compute_score/predict.")
        except Exception as exc:
            raise RuntimeError("Reranker failed to score pairs.") from exc

        if not isinstance(scores, list):
            scores = list(scores)
        
        raw_scores = [float(s) for s in scores]
        
        reranked = [
            {**item, "rerank_score": float(score)}
            for item, score in zip(items, scores)
        ]
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked, raw_scores

    def search(
        self,
        query: str,
        *,
        top_k_fused: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        top_k_final: Optional[int] = None,
        source_id: Optional[str] = None,
        collect_metrics: bool = True,
    ) -> list[RetrievalResult]:
        """Execute hybrid search with timing and metrics collection.

        The retrieval pipeline is now:
        1. LanceDB hybrid search (vector ANN + FTS BM25 w/ RRF) → top_k_fused
        2. Deduplicate by parent
        3. Enrich with parent text for reranking
        4. Rerank (Jina v3) → threshold filter → top_k_final
        """
        cfg = self._config
        k_fused = top_k_fused or (cfg.top_k_fused if cfg else 50)
        k_rerank = top_k_rerank or (cfg.top_k_rerank if cfg else 20)
        k_final = top_k_final or (cfg.top_k_final if cfg else 5)

        timing = TimingMetrics()
        total_start = time.perf_counter()

        # Stage 1: LanceDB native hybrid search (replaces dense + sparse + RRF)
        t0 = time.perf_counter()
        fused = self._hybrid_search(query, k_fused, source_id=source_id)
        timing.hybrid_search_ms = (time.perf_counter() - t0) * 1000
        # sparse_search_ms and rrf_fusion_ms are zero — LanceDB does it all in one call
        timing.sparse_search_ms = 0.0
        timing.rrf_fusion_ms = 0.0
        logger.info("LanceDB hybrid search returned %d hits in %.3fms", len(fused), timing.hybrid_search_ms)

        # Stage 2: Deduplicate by parent
        t0 = time.perf_counter()
        deduped, dedup_metrics = self._deduplicate_by_parent(fused, k_fused)
        timing.dedup_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Early dedup: {dedup_metrics.children_before_dedup} -> {dedup_metrics.children_after_dedup} ({dedup_metrics.reduction_pct:.1f}% reduction)")

        # Stage 3: Enrich with parent text for reranking
        parent_ids = {
            metadata.get("parent_id")
            for metadata in (item.get("metadata") or {} for item in deduped)
            if isinstance(metadata.get("parent_id"), str) and metadata.get("parent_id")
        }
        parent_cache = self._storage.get_parent_texts(parent_ids)
        for item in deduped:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if isinstance(parent_id, str) and parent_id in parent_cache:
                item["rerank_text"] = parent_cache[parent_id]

        # Stage 4: Rerank
        t0 = time.perf_counter()
        to_rerank = deduped[:k_rerank]
        reranked, raw_scores = self._rerank(query, to_rerank)
        timing.rerank_ms = (time.perf_counter() - t0) * 1000

        # Stage 5: Threshold filtering
        threshold_metrics = ThresholdMetrics()
        if cfg and reranked:
            threshold = float(cfg.reranker_threshold)
            min_docs = cfg.reranker_min_docs
            items_before = len(reranked)

            threshold_filtered = [
                item for item in reranked
                if item.get("rerank_score", float("-inf")) >= threshold
            ]

            if len(threshold_filtered) < min_docs:
                logger.debug(
                    f"Threshold filter produced {len(threshold_filtered)} docs "
                    f"(< min {min_docs}), using top {min_docs} reranked results"
                )
                reranked = reranked[:min_docs]
                threshold_metrics = ThresholdMetrics(
                    threshold_value=threshold,
                    items_before_threshold=items_before,
                    items_after_threshold=min_docs,
                    safety_net_triggered=True,
                    min_docs=min_docs,
                )
            else:
                logger.debug(
                    "Threshold filter: %d -> %d docs (threshold: %.4f)",
                    len(reranked),
                    len(threshold_filtered),
                    threshold,
                )
                reranked = threshold_filtered
                threshold_metrics = ThresholdMetrics(
                    threshold_value=threshold, items_before_threshold=items_before,
                    items_after_threshold=len(threshold_filtered), safety_net_triggered=False, min_docs=min_docs,
                )

        reranker_metrics = compute_reranker_stats(raw_scores)
        if reranker_metrics.items_reranked > 0:
            logger.debug(
                "Rerank score stats: min=%.4f max=%.4f mean=%.4f std=%.4f (n=%d)",
                reranker_metrics.score_min,
                reranker_metrics.score_max,
                reranker_metrics.score_mean,
                reranker_metrics.score_std,
                reranker_metrics.items_reranked,
            )

        # Stage 6: Final dedup + boilerplate filter
        final: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        seen_children: set[str] = set()
        for item in reranked:
            text_for_filter = (item.get("rerank_text") or item.get("text") or "").lower()
            if any(pat in text_for_filter for pat in _BOILERPLATE_PATTERNS):
                continue
            child_id = item.get("id")
            if isinstance(child_id, str):
                if child_id in seen_children:
                    continue
                seen_children.add(child_id)
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if isinstance(parent_id, str):
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
            final.append(item)
            if len(final) >= k_final:
                break

        final_parent_ids = {
            metadata.get("parent_id")
            for metadata in (item.get("metadata") or {} for item in final)
            if isinstance(metadata.get("parent_id"), str) and metadata.get("parent_id")
        }
        missing_parent_ids = [pid for pid in final_parent_ids if pid not in parent_cache]
        if missing_parent_ids:
            parent_cache.update(self._storage.get_parent_texts(missing_parent_ids))

        results: list[RetrievalResult] = []
        for item in final:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            parent_text = parent_cache.get(parent_id) if isinstance(parent_id, str) else None
            results.append(RetrievalResult(
                child_id=item["id"], text=item.get("text", ""), metadata=metadata,
                score=float(item.get("rerank_score", item.get("score", 0.0))), parent_text=parent_text,
            ))

        timing.total_ms = (time.perf_counter() - total_start) * 1000

        if collect_metrics and results:
            metrics = RetrievalMetrics(
                timing=timing, reranker=reranker_metrics, deduplication=dedup_metrics,
                threshold=threshold_metrics, query=query, mode=cfg.mode if cfg else "unknown",
            )
            results[0] = RetrievalResult(
                child_id=results[0].child_id, text=results[0].text, metadata=results[0].metadata,
                score=results[0].score, parent_text=results[0].parent_text, metrics=metrics,
            )

        return results
