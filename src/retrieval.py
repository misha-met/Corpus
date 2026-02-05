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
    
    Creates a structured chunk format that enables the LLM to cite sources:
    [CHUNK START | SOURCE: source_id | PAGE: display_page]
    ... chunk content ...
    [CHUNK END]
    
    Args:
        text: The chunk text content
        source_id: Source document identifier (becomes SourceID in citations)
        display_page: Human-readable page (e.g., 'iii', '42')
        chunk_index: Sequential index for this chunk (for reference)
    
    Returns:
        Formatted chunk string with citation markers
    """
    page_info = f" | PAGE: {display_page}" if display_page else ""
    header = f"[CHUNK START | SOURCE: {source_id}{page_info}]"
    footer = "[CHUNK END]"
    return f"{header}\n{text.strip()}\n{footer}"


def format_context_with_citations(
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Format multiple chunks with citation markers and build source mapping.
    
    Args:
        texts: List of chunk text contents
        metadatas: List of metadata dicts (must have 'source_id', optionally 'display_page')
    
    Returns:
        Tuple of:
        - Formatted context string with all chunks
        - Source ID to document name mapping (for citation legend, only non-trivial mappings)
    
    Raises:
        ValueError: If texts and metadatas have mismatched lengths
    """
    # Validate input lengths
    if len(texts) != len(metadatas):
        raise ValueError(
            f"texts and metadatas must have same length, got {len(texts)} vs {len(metadatas)}"
        )
    
    formatted_chunks: list[str] = []
    source_mapping: dict[str, str] = {}
    
    for idx, (text, meta) in enumerate(zip(texts, metadatas), start=1):
        source_id = meta.get("source_id", "Unknown")
        display_page = meta.get("display_page")
        
        # Build source mapping only when doc_name differs from source_id
        # This avoids redundant mappings like "Source 1 → Source 1"
        doc_name = meta.get("doc_name") or meta.get("filename")
        if doc_name and doc_name != source_id and source_id not in source_mapping:
            source_mapping[source_id] = doc_name
        
        formatted_chunk = format_chunk_for_citation(
            text=text,
            source_id=source_id,
            display_page=display_page,
            chunk_index=idx,
        )
        formatted_chunks.append(formatted_chunk)
    
    context = "\n\n".join(formatted_chunks)
    return context, source_mapping


def build_source_legend(source_mapping: dict[str, str]) -> str:
    """Build a source legend for citation reference.
    
    Creates a mapping list that helps readers identify sources.
    Only includes entries where source_id differs from document name,
    to avoid redundant mappings that provide no utility.
    
    SOURCE LEGEND:
    - SourceID1 → Document Name 1
    - SourceID2 → Document Name 2
    
    Args:
        source_mapping: Dict of source_id -> document name (non-trivial mappings only)
    
    Returns:
        Formatted source legend string, or empty string if no useful mappings exist
    """
    # Filter out any remaining trivial mappings (source_id == doc_name)
    useful_mappings = {
        sid: name for sid, name in source_mapping.items() if sid != name
    }
    
    if not useful_mappings:
        return ""
    
    lines = ["SOURCE LEGEND:"]
    for source_id, doc_name in sorted(useful_mappings.items()):
        lines.append(f"- {source_id} → {doc_name}")
    
    return "\n".join(lines)


@dataclass(frozen=True)
class RetrievalResult:
    """Result from retrieval pipeline.
    
    Attributes:
        child_id: Unique identifier of the child chunk
        text: Text content of the child chunk
        metadata: Chunk metadata (source_id, parent_id, etc.)
        score: Final score (rerank score if available, else RRF score)
        parent_text: Full text of the parent chunk (for context expansion)
        metrics: Optional retrieval metrics (attached to first result only)
    """
    child_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    parent_text: Optional[str]
    metrics: Optional[RetrievalMetrics] = None


class RetrievalEngine:
    """Hybrid retrieval engine with dense, sparse, and reranking stages.
    
    Implements the full retrieval pipeline:
    1. Parallel dense (embedding) and sparse (BM25) search
    2. RRF fusion of results
    3. Early parent-level deduplication
    4. Reranking with score distribution tracking
    5. Context expansion to parent chunks
    
    Attributes:
        config: ModelConfig with retrieval parameters (optional, uses defaults)
    """
    
    def __init__(
        self,
        *,
        storage: StorageEngine,
        embedding_model: Any,
        reranker: Optional[Any] = None,
        config: Optional[ModelConfig] = None,
    ) -> None:
        """Initialize the retrieval engine.
        
        Args:
            storage: StorageEngine for vector/BM25/SQLite access
            embedding_model: SentenceTransformer model for embeddings
            reranker: Optional FlagReranker for reranking
            config: Optional ModelConfig for retrieval parameters
        """
        self._storage = storage
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._config = config

    def _dense_search(
        self,
        query: str,
        top_k: int,
        *,
        source_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        try:
            embeddings = self._embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("Embedding model encode failed.") from exc

        where = {"source_id": source_id} if source_id else None
        response = self._storage.query_children(
            embeddings=embeddings,
            top_k=top_k,
            where=where,
        )

        results: list[dict[str, Any]] = []
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances),
            start=1,
        ):
            results.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": meta or {},
                    "rank": rank,
                    "distance": dist,
                }
            )

        return results

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        *,
        source_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        bm25 = self._storage.bm25
        if bm25 is None:
            raise RuntimeError("BM25 index is not initialized.")

        tokenized = query.split()
        if not tokenized:
            raise ValueError("query must contain tokens for BM25 search.")

        scores = bm25.get_scores(tokenized)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        source_ids = self._storage.bm25_source_ids
        source_filter = bool(source_id)
        if source_filter and (not source_ids or len(source_ids) != len(self._storage.bm25_ids)):
            raise RuntimeError(
                "BM25 index source_ids are missing or misaligned; "
                "rebuild the BM25 index to use source_id filtering."
            )

        results: list[dict[str, Any]] = []
        rank = 1
        for idx, score in ranked:
            child_id = self._storage.bm25_ids[idx]
            if source_filter:
                if source_ids[idx] != source_id:
                    continue
            results.append(
                {
                    "id": child_id,
                    "score": float(score),
                    "rank": rank,
                }
            )
            rank += 1
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _rrf_fuse(
        dense: Iterable[dict[str, Any]],
        sparse: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for item in dense:
            rank = item["rank"]
            score = 1.0 / (60 + rank)
            scores[item["id"]] = scores.get(item["id"], 0.0) + score
            payloads[item["id"]] = item

        for item in sparse:
            rank = item["rank"]
            score = 1.0 / (60 + rank)
            scores[item["id"]] = scores.get(item["id"], 0.0) + score
            payloads.setdefault(item["id"], item)

        fused = [
            {"id": cid, "score": score, **payloads.get(cid, {})}
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda item: item["score"], reverse=True)
        return fused

    @staticmethod
    def _deduplicate_by_parent(
        items: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], DeduplicationMetrics]:
        """Early deduplication: keep highest-scored child per unique parent_id.
        
        This runs BEFORE reranking to reduce redundant rerank computations.
        For each parent_id, only the child with the highest RRF score is kept.
        
        Args:
            items: List of fused results with metadata
            top_k: Maximum number of results to return
            
        Returns:
            Tuple of (deduplicated items, deduplication metrics)
        """
        before_count = len(items)
        seen_parents: dict[str, dict[str, Any]] = {}
        no_parent_items: list[dict[str, Any]] = []
        
        for item in items:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            
            if not isinstance(parent_id, str):
                # Items without parent_id are kept separately
                no_parent_items.append(item)
                continue
            
            # Keep highest-scored item per parent
            if parent_id not in seen_parents:
                seen_parents[parent_id] = item
            elif item.get("score", 0) > seen_parents[parent_id].get("score", 0):
                seen_parents[parent_id] = item
        
        # Combine and sort by score
        deduplicated = list(seen_parents.values()) + no_parent_items
        deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)
        deduplicated = deduplicated[:top_k]
        
        after_count = len(deduplicated)
        parents_removed = before_count - after_count - len(no_parent_items) + len([
            i for i in deduplicated if not (i.get("metadata") or {}).get("parent_id")
        ])
        
        metrics = DeduplicationMetrics(
            children_before_dedup=before_count,
            children_after_dedup=after_count,
            reduction_pct=100 * (1 - after_count / before_count) if before_count > 0 else 0,
            parents_deduplicated=max(0, before_count - after_count),
        )
        
        return deduplicated, metrics

    def _rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Rerank items and return both reranked items and raw scores.
        
        Args:
            query: The search query
            items: Items to rerank (should have 'rerank_text' or 'text')
            
        Returns:
            Tuple of (reranked items, raw scores before penalty)
        """
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
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("Reranker failed to score pairs.") from exc

        # Ensure scores is a list
        if not isinstance(scores, list):
            scores = list(scores)
        
        raw_scores = [float(s) for s in scores]
        
        reranked = []
        for item, score in zip(items, scores):
            text = (item.get("rerank_text") or item.get("text") or "").lower()
            penalty = 0.5 if any(pat in text for pat in _BOILERPLATE_PATTERNS) else 0.0
            reranked.append({**item, "rerank_score": float(score) - penalty})
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked, raw_scores

    def search(
        self,
        query: str,
        *,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_fused: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        top_k_final: Optional[int] = None,
        source_id: Optional[str] = None,
        collect_metrics: bool = True,
    ) -> list[RetrievalResult]:
        """Execute hybrid search with timing and metrics collection.
        
        Args:
            query: Search query string
            top_k_dense: Dense search results (default from config or 100)
            top_k_sparse: Sparse search results (default from config or 100)
            top_k_fused: Results after RRF fusion (default from config or 50)
            top_k_rerank: Results to pass to reranker (default from config or 20)
            top_k_final: Final results to return (default from config or 5)
            source_id: Optional source filter
            collect_metrics: Whether to collect and attach metrics
            
        Returns:
            List of RetrievalResult with metrics attached to first result
        """
        # Resolve parameters from config or defaults
        cfg = self._config
        k_dense = top_k_dense or (cfg.top_k_dense if cfg else 100)
        k_sparse = top_k_sparse or (cfg.top_k_sparse if cfg else 100)
        k_fused = top_k_fused or (cfg.top_k_fused if cfg else 50)
        k_rerank = top_k_rerank or (cfg.top_k_rerank if cfg else 20)
        k_final = top_k_final or (cfg.top_k_final if cfg else 5)
        
        # Initialize timing
        timing = TimingMetrics()
        total_start = time.perf_counter()
        
        # Stage 1: Dense search
        t0 = time.perf_counter()
        dense = self._dense_search(query, k_dense, source_id=source_id)
        timing.dense_search_ms = (time.perf_counter() - t0) * 1000
        
        # Stage 2: Sparse search
        t0 = time.perf_counter()
        sparse = self._sparse_search(query, k_sparse, source_id=source_id)
        timing.sparse_search_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Sparse search returned %s hits",
            len(sparse),
        )
        
        # Stage 3: RRF fusion
        t0 = time.perf_counter()
        fused = self._rrf_fuse(dense, sparse)
        timing.rrf_fusion_ms = (time.perf_counter() - t0) * 1000
        
        # Fetch missing text/metadata
        missing_ids = [
            item["id"]
            for item in fused
            if "text" not in item or "metadata" not in item
        ]
        if missing_ids:
            fetched = self._storage.get_children_by_ids(missing_ids)
            for item in fused:
                if item["id"] in fetched:
                    item.setdefault("text", fetched[item["id"]].get("text"))
                    item.setdefault("metadata", fetched[item["id"]].get("metadata"))

        for item in fused:
            if "text" not in item or "metadata" not in item:
                lookup = next(
                    (d for d in dense if d["id"] == item["id"]),
                    None,
                )
                if lookup:
                    item.setdefault("text", lookup.get("text"))
                    item.setdefault("metadata", lookup.get("metadata"))

        # Stage 4: Early deduplication (before rerank)
        t0 = time.perf_counter()
        deduped, dedup_metrics = self._deduplicate_by_parent(fused, k_fused)
        timing.dedup_ms = (time.perf_counter() - t0) * 1000
        
        logger.debug(
            f"Early dedup: {dedup_metrics.children_before_dedup} -> "
            f"{dedup_metrics.children_after_dedup} ({dedup_metrics.reduction_pct:.1f}% reduction)"
        )
        
        # Prepare rerank text (parent context)
        parent_cache: dict[str, str] = {}
        for item in deduped:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if isinstance(parent_id, str):
                if parent_id not in parent_cache:
                    parent_text = self._storage.get_parent_text(parent_id)
                    if parent_text:
                        parent_cache[parent_id] = parent_text
                if parent_id in parent_cache:
                    item["rerank_text"] = parent_cache[parent_id]

        # Stage 5: Reranking
        t0 = time.perf_counter()
        to_rerank = deduped[:k_rerank]
        reranked, raw_scores = self._rerank(query, to_rerank)
        timing.rerank_ms = (time.perf_counter() - t0) * 1000
        
        # Apply threshold filtering with safety net
        threshold_metrics = ThresholdMetrics()
        if cfg and reranked:
            threshold = cfg.reranker_threshold
            min_docs = cfg.reranker_min_docs
            items_before = len(reranked)
            
            # Filter by threshold
            threshold_filtered = [
                item for item in reranked 
                if item.get("rerank_score", float("-inf")) >= threshold
            ]
            
            # Apply safety net: ensure minimum documents
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
                    f"Threshold filter: {len(reranked)} -> {len(threshold_filtered)} docs "
                    f"(threshold: {threshold:.1f})"
                )
                reranked = threshold_filtered
                threshold_metrics = ThresholdMetrics(
                    threshold_value=threshold,
                    items_before_threshold=items_before,
                    items_after_threshold=len(threshold_filtered),
                    safety_net_triggered=False,
                    min_docs=min_docs,
                )
        
        # Compute reranker score statistics
        reranker_metrics = compute_reranker_stats(raw_scores)
        
        # Stage 6: Final selection with boilerplate filtering
        final: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        seen_children: set[str] = set()
        for item in reranked:
            text_for_filter = (
                item.get("rerank_text")
                or item.get("text")
                or ""
            ).lower()
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

        # Build results
        results: list[RetrievalResult] = []
        for item in final:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            parent_text = (
                self._storage.get_parent_text(parent_id)
                if isinstance(parent_id, str)
                else None
            )
            results.append(
                RetrievalResult(
                    child_id=item["id"],
                    text=item.get("text", ""),
                    metadata=metadata,
                    score=float(item.get("rerank_score", item.get("score", 0.0))),
                    parent_text=parent_text,
                )
            )

        # Finalize timing
        timing.total_ms = (time.perf_counter() - total_start) * 1000
        
        # Build metrics
        if collect_metrics and results:
            metrics = RetrievalMetrics(
                timing=timing,
                reranker=reranker_metrics,
                deduplication=dedup_metrics,
                threshold=threshold_metrics,
                query=query,
                mode=cfg.mode if cfg else "unknown",
            )
            # Attach metrics to first result
            results[0] = RetrievalResult(
                child_id=results[0].child_id,
                text=results[0].text,
                metadata=results[0].metadata,
                score=results[0].score,
                parent_text=results[0].parent_text,
                metrics=metrics,
            )
        
        return results
