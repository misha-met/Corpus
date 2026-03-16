"""Hybrid retrieval engine: vector + BM25 search → reranking → filtering → expansion.

Architecture
~~~~~~~~~~~~
- The pipeline is broken into discrete stage methods so each step can be
  timed, tested, and overridden independently:
    1. ``_stage_hybrid_search()``   — LanceDB native hybrid search + query embed
    2. ``_stage_rerank()``          — Jina reranker v3 listwise scoring
    3. ``_stage_threshold_filter()``— adaptive score threshold + safety floor
    4. ``_stage_budget_expand()``   — deduplication and top-k trimming
    5. ``_stage_context_expand()``  — fetch parent chunk text for LLM context
- ``_hybrid_search_decoupled()`` is the lower-level method that separates the
  embedding query from the BM25 query, allowing the caller to pre-compute the
  query vector and pass it in to avoid redundant encoding.
- Sub-threshold policy (how many below-threshold results to retain) varies by
  intent and query type to balance precision against recall.
"""
from __future__ import annotations

import heapq
import logging
import re
import time
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Optional

from .config import ModelConfig, ResolvedRetrievalParams
from .metrics import (
    RetrievalMetrics,
    TimingMetrics,
    RerankerMetrics,
    DeduplicationMetrics,
    ThresholdMetrics,
    compute_reranker_stats,
)
from .phoenix_tracing import (
    SPAN_KIND_RERANKER,
    SPAN_KIND_RETRIEVER,
    format_openinference_document,
    set_reranker_documents,
    set_retrieval_documents,
    set_span_attributes,
    start_span,
)
from .storage import StorageEngine

logger = logging.getLogger(__name__)

# Approximate subword expansion: BPE tokenizers produce ~1.35 tokens per whitespace word.
# Applied to retrieval budget estimates only. Do NOT change ingest chunking tokenizer.
_WORD_TO_TOKEN_RATIO: float = 1.35

# Adaptive reranker threshold: proportion of the top score below which
# candidates are discarded. Lower values keep more marginal results.
_ADAPTIVE_THRESHOLD_FACTOR: float = 0.15

_PAGE_MARKER_RE = re.compile(r"\[Page\s+\d+\]", re.IGNORECASE)


@dataclass(frozen=True)
class _SubThresholdPolicy:
    starvation_floor_ratio: float
    budget_ceiling_ratio: float
    max_additional_chunks: int


_SUB_THRESHOLD_ENUMERATION = _SubThresholdPolicy(
    starvation_floor_ratio=0.10,
    budget_ceiling_ratio=0.15,
    max_additional_chunks=25,
)
_SUB_THRESHOLD_ANALYTICAL = _SubThresholdPolicy(
    starvation_floor_ratio=0.10,
    budget_ceiling_ratio=0.12,
    max_additional_chunks=10,
)
_SUB_THRESHOLD_FACTUAL = _SubThresholdPolicy(
    starvation_floor_ratio=0.08,
    budget_ceiling_ratio=0.09,
    max_additional_chunks=4,
)

_ENUMERATION_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blist\s+(all|every)\b", re.IGNORECASE),
    re.compile(r"\bfind\s+(all|every)\b", re.IGNORECASE),
    re.compile(r"\bname\s+all\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+mentions?\s+of\b", re.IGNORECASE),
    re.compile(r"\bmentions?\s+of\b.+\bare\s+there\b", re.IGNORECASE),
    re.compile(r"\breferences?\s+(to|of)\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+is\b.+\bmentioned\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\s+times\s+is\b.+\bmentioned\b", re.IGNORECASE),
)


def _is_enumeration_query(query: str) -> bool:
    text = query.strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _ENUMERATION_QUERY_PATTERNS)


def _resolve_sub_threshold_policy(
    *,
    query: str,
    intent: Optional[str],
) -> tuple[str, _SubThresholdPolicy]:
    if _is_enumeration_query(query):
        return "enumeration", _SUB_THRESHOLD_ENUMERATION

    intent_key = (intent or "").strip().upper()
    if intent_key == "FACTUAL":
        return "factual", _SUB_THRESHOLD_FACTUAL
    if intent_key in {"ANALYZE", "COMPARE", "CRITIQUE", "SUMMARIZE", "OVERVIEW", "EXPLAIN"}:
        return "analytical", _SUB_THRESHOLD_ANALYTICAL
    return "default", _SUB_THRESHOLD_ANALYTICAL


def format_chunk_for_citation(
    text: str,
    source_id: str,
    display_page: Optional[str] = None,
    page_number: Optional[int] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    chunk_index: int = 1,
) -> str:
    """Format a retrieved passage with citation markers for Academic Mode.

    Passages are numbered sequentially so the LLM can cite by number [1], [2].
    The block header always includes source and, when available, page and chunk
    metadata to reduce page attribution ambiguity.
    """
    page_label: Optional[str] = None
    if isinstance(start_page, int) and isinstance(end_page, int) and start_page != end_page:
        page_label = f"{start_page}-{end_page}"
    elif display_page:
        page_label = str(display_page).strip()
    elif isinstance(page_number, int):
        page_label = str(page_number)
    elif isinstance(start_page, int):
        page_label = str(start_page)

    marker_page: Optional[int] = None
    if isinstance(page_number, int):
        marker_page = page_number
    elif isinstance(start_page, int):
        marker_page = start_page
    elif display_page:
        try:
            marker_page = int(str(display_page).strip())
        except (TypeError, ValueError):
            marker_page = None

    body = text.strip()
    if marker_page is not None and body and not _PAGE_MARKER_RE.search(body):
        body = f"[Page {marker_page}]\n{body}"

    block_parts = [f"Source: {source_id}"]
    if page_label:
        block_parts.append(f"Page {page_label}")
    if chunk_id:
        block_parts.append(f"Chunk {chunk_id}")
    source_block_header = " | ".join(block_parts)

    return (
        f"[PASSAGE {chunk_index}]\n"
        f"[{source_block_header}]\n"
        f"{body}\n"
        f"[/Source]\n"
        f"[PASSAGE END]"
    )


def format_context_with_citations(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    chunk_ids: Optional[list[str]] = None,
) -> tuple[str, dict[str, str]]:
    """Format passages with citation markers and build source mapping."""
    if len(texts) != len(metadatas):
        raise ValueError(f"texts and metadatas must have same length, got {len(texts)} vs {len(metadatas)}")
    if chunk_ids is not None and len(chunk_ids) != len(texts):
        raise ValueError(f"chunk_ids must have same length as texts, got {len(chunk_ids)} vs {len(texts)}")

    formatted_chunks: list[str] = []
    source_mapping: dict[str, str] = {}
    
    for idx, (text, meta) in enumerate(zip(texts, metadatas), start=1):
        source_id = meta.get("source_id", "Unknown")
        display_page = meta.get("display_page")
        page_number = meta.get("page_number")
        start_page = meta.get("start_page")
        end_page = meta.get("end_page")

        resolved_chunk_id: Optional[str] = None
        if chunk_ids is not None:
            value = chunk_ids[idx - 1]
            if isinstance(value, str) and value:
                resolved_chunk_id = value
        if resolved_chunk_id is None:
            maybe_chunk_id = meta.get("chunk_id")
            if isinstance(maybe_chunk_id, str) and maybe_chunk_id:
                resolved_chunk_id = maybe_chunk_id

        doc_name = meta.get("doc_name") or meta.get("filename")
        if doc_name and doc_name != source_id and source_id not in source_mapping:
            source_mapping[source_id] = doc_name

        formatted_chunks.append(format_chunk_for_citation(
            text=text,
            source_id=source_id,
            display_page=display_page,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
            chunk_id=resolved_chunk_id,
            chunk_index=idx,
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
        tracer: Optional[Any] = None,
    ) -> None:
        self._storage = storage
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._config = config
        self._tracer = tracer
        # NOTE: This engine is instantiated per-request (see rag_engine.py query() /
        # _query_events_impl()), so no query embedding cache is needed.  If a
        # long-lived engine is introduced later, consider adding a cache keyed on
        # (model_id, query.strip()) with an LRU eviction policy.

    @staticmethod
    def _encode_query(embedding_model: Any, query: str) -> list[float]:
        """Encode a single query string and return a normalised float vector."""
        try:
            embeddings = embedding_model.encode([query], normalize_embeddings=True)
            embedding = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else list(embeddings[0])
            return [float(v) for v in embedding]
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("Embedding model encode failed.") from exc

    def _hybrid_search_decoupled(
        self,
        *,
        embedding_query: str,
        bm25_query: str,
        top_k: int,
        source_id: Optional[str] = None,
        query_vector: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search with separate queries for embedding and BM25.

        When *query_vector* is supplied the embedding encode step is skipped;
        the caller is responsible for timing that encode separately.
        """
        if not embedding_query.strip():
            raise ValueError("embedding_query must be a non-empty string.")
        if not bm25_query.strip():
            raise ValueError("bm25_query must be a non-empty string.")
        if query_vector is None:
            query_vector = self._encode_query(self._embedding_model, embedding_query)

        return self._storage.hybrid_search(
            query_text=bm25_query,
            query_vector=query_vector,
            top_k=top_k,
            source_id=source_id,
        )

    @staticmethod
    def _deduplicate_by_parent(
        items: list[dict[str, Any]],
        top_k: int,
        max_children_per_parent: int = 2,
    ) -> tuple[list[dict[str, Any]], DeduplicationMetrics]:
        """Keep up to *max_children_per_parent* highest-scored children per parent_id.

        The result is trimmed to *top_k* by score so the caller's budget is
        unchanged.  The default of 2 sends a richer candidate set to the
        reranker while still avoiding the long-tail of low-scoring siblings.
        Pass ``max_children_per_parent=1`` for the original one-per-parent
        behaviour.
        """
        before_count = len(items)
        parent_children: dict[str, list[dict[str, Any]]] = {}
        no_parent_items: list[dict[str, Any]] = []

        for item in items:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            parent_id = metadata.get("parent_id")

            if not isinstance(parent_id, str):
                no_parent_items.append(item)
                continue

            children = parent_children.setdefault(parent_id, [])
            if len(children) < max_children_per_parent:
                children.append(item)
                children.sort(key=lambda x: x.get("score", 0), reverse=True)
            elif item.get("score", 0) > children[-1].get("score", 0):
                children[-1] = item
                children.sort(key=lambda x: x.get("score", 0), reverse=True)

        multi_parent_count = sum(1 for ch in parent_children.values() if len(ch) > 1)
        extra_children = sum(len(ch) - 1 for ch in parent_children.values() if len(ch) > 1)
        if multi_parent_count > 0:
            logger.debug(
                "Dedup (max_per_parent=%d): %d parents had >1 child; %d extra children "
                "retained vs one-per-parent policy",
                max_children_per_parent, multi_parent_count, extra_children,
            )

        deduplicated = [
            item for children in parent_children.values() for item in children
        ] + no_parent_items
        deduplicated_count = len(deduplicated)
        dedup_removed_count = max(0, before_count - deduplicated_count)
        top_k_limited = heapq.nlargest(top_k, deduplicated, key=itemgetter("score"))

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
            {**item, "rerank_score": float(score), "score": float(score)}
            for item, score in zip(items, scores)
        ]
        reranked.sort(key=itemgetter("rerank_score"), reverse=True)
        return reranked, raw_scores

    def _stage_hybrid_search(
        self,
        query: str,
        embedding_query: str,
        bm25_query: str,
        k_fused: int,
        source_id: Optional[str],
        timing: "TimingMetrics",
    ) -> tuple[list[dict[str, Any]], Any]:
        """Stage 1: LanceDB native hybrid search + query embedding."""
        t0 = time.perf_counter()
        query_vector = self._encode_query(self._embedding_model, embedding_query)
        timing.query_embedding_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = self._hybrid_search_decoupled(
            embedding_query=embedding_query,
            bm25_query=bm25_query,
            top_k=k_fused,
            source_id=source_id,
            query_vector=query_vector,
        )
        timing.hybrid_search_ms = (time.perf_counter() - t0) * 1000
        timing.sparse_search_ms = 0.0
        timing.rrf_fusion_ms = 0.0
        logger.info("LanceDB hybrid search returned %d hits in %.3fms", len(fused), timing.hybrid_search_ms)
        return fused, query_vector

    def _stage_rerank(
        self,
        query: str,
        fused: list[dict[str, Any]],
        k_rerank: int,
        reranker_enabled: bool,
        timing: "TimingMetrics",
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Stage 2: Rerank child chunks."""
        t0 = time.perf_counter()
        if reranker_enabled:
            to_rerank = fused[:k_rerank]
            reranked, raw_scores = self._rerank(query, to_rerank)
        else:
            reranked = fused
            raw_scores = []
        timing.rerank_ms = (time.perf_counter() - t0) * 1000
        return reranked, raw_scores

    def _stage_threshold_filter(
        self,
        reranked: list[dict[str, Any]],
        reranker_enabled: bool,
        reranker_threshold: float,
        reranker_min_docs: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, "ThresholdMetrics"]:
        """Stage 4: Adaptive threshold filtering.

        Returns (filtered, above_threshold_results, effective_threshold, metrics).
        """
        threshold_metrics = ThresholdMetrics()
        threshold = float(reranker_threshold)
        above_threshold_results: list[dict[str, Any]] = []

        if not reranked:
            return reranked, above_threshold_results, threshold, threshold_metrics

        config_threshold = reranker_threshold
        min_docs = reranker_min_docs
        items_before = len(reranked)
        score_key = "rerank_score" if reranker_enabled else "score"

        top_score = float(reranked[0].get(score_key, 0.0))
        relative_factor = _ADAPTIVE_THRESHOLD_FACTOR
        adaptive_threshold = max(config_threshold, top_score * relative_factor)
        logger.info(
            "Adaptive threshold: top_score=%.4f relative=%.4f effective=%.4f (config floor=%.4f)",
            top_score, top_score * relative_factor, adaptive_threshold, config_threshold,
        )
        threshold = adaptive_threshold

        threshold_filtered = [
            item for item in reranked
            if float(item.get(score_key, float("-inf"))) >= threshold
        ]
        above_threshold_results = list(threshold_filtered)

        if len(threshold_filtered) < min_docs:
            logger.debug(
                f"Threshold filter produced {len(threshold_filtered)} docs "
                f"(< min {min_docs}), using top {min_docs} reranked results"
            )
            reranked = reranked[:min_docs]
            for item in reranked:
                if float(item.get(score_key, float("-inf"))) < threshold:
                    item["below_threshold"] = True
                    meta = item.get("metadata")
                    if isinstance(meta, dict):
                        meta["below_threshold"] = True
            threshold_metrics = ThresholdMetrics(
                threshold_value=threshold, items_before_threshold=items_before,
                items_after_threshold=min_docs, safety_net_triggered=True, min_docs=min_docs,
            )
        else:
            logger.debug(
                "Threshold filter: %d -> %d docs (threshold: %.4f)",
                len(reranked), len(threshold_filtered), threshold,
            )
            reranked = threshold_filtered
            threshold_metrics = ThresholdMetrics(
                threshold_value=threshold, items_before_threshold=items_before,
                items_after_threshold=len(threshold_filtered), safety_net_triggered=False, min_docs=min_docs,
            )

        return reranked, above_threshold_results, threshold, threshold_metrics

    def _stage_budget_expand(
        self,
        query: str,
        reranked: list[dict[str, Any]],
        all_reranked: list[dict[str, Any]],
        above_threshold_results: list[dict[str, Any]],
        k_final: int,
        budget: int,
        threshold: float,
        intent: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Stage 5: Budget-aware expansion and sub-threshold backfill.

        Returns (expanded_reranked, new_k_final).
        """
        if budget <= 0 or not reranked:
            return reranked, k_final

        policy_name, sub_threshold_policy = _resolve_sub_threshold_policy(query=query, intent=intent)

        def _est_tokens(text: str) -> int:
            return int(len(text.split()) * _WORD_TO_TOKEN_RATIO)

        candidates_for_final = reranked[:k_final]
        selected_tokens = sum(_est_tokens(item.get("text", "")) for item in candidates_for_final)
        budget_floor = int(budget * 0.25)

        if selected_tokens < budget_floor and len(reranked) > k_final:
            budget_ceiling = int(budget * 0.50)
            base_count = k_final
            running_tokens = selected_tokens
            expanded_k = k_final
            for item in reranked[k_final:]:
                item_tokens = _est_tokens(item.get("text", ""))
                if running_tokens + item_tokens > budget_ceiling:
                    break
                running_tokens += item_tokens
                expanded_k += 1
            if expanded_k > k_final:
                pct = round(100 * running_tokens / budget) if budget else 0
                logger.info("Budget-aware expansion: %d -> %d chunks (%d%% of budget)", base_count, expanded_k, pct)
                k_final = expanded_k

        # Sub-threshold expansion for severely starved budgets
        current_final_tokens = sum(_est_tokens(item.get("text", "")) for item in reranked[:k_final])
        starvation_floor = int(budget * sub_threshold_policy.starvation_floor_ratio)

        if current_final_tokens < starvation_floor:
            threshold_ids = {item.get("id") for item in reranked}

            def _candidate_source_id(item: dict[str, Any]) -> Optional[str]:
                metadata = item.get("metadata")
                if isinstance(metadata, dict):
                    source = metadata.get("source_id")
                    if isinstance(source, str) and source:
                        return source
                source = item.get("source_id")
                if isinstance(source, str) and source:
                    return source
                return None

            sources_with_hits = {
                source
                for source in (_candidate_source_id(item) for item in above_threshold_results)
                if source
            }
            sub_threshold_candidates = [
                item for item in all_reranked
                if item.get("id") not in threshold_ids
                and (not sources_with_hits or _candidate_source_id(item) in sources_with_hits)
            ]
            sub_ceiling = int(budget * sub_threshold_policy.budget_ceiling_ratio)
            running_sub = current_final_tokens
            sub_added = 0
            for item in sub_threshold_candidates:
                if sub_added >= sub_threshold_policy.max_additional_chunks:
                    break
                item_tok = _est_tokens(item.get("text", ""))
                if running_sub + item_tok > sub_ceiling:
                    break
                meta = item.get("metadata")
                if isinstance(meta, dict):
                    meta["below_threshold"] = True
                item["below_threshold"] = True
                reranked.append(item)
                running_sub += item_tok
                sub_added += 1
                k_final += 1
            if sub_added > 0:
                sub_pct = round(100 * running_sub / budget) if budget else 0
                logger.info(
                    "Sub-threshold expansion (%s): added %d chunks below threshold=%.4f (budget now %d%%, cap=%d)",
                    policy_name, sub_added, threshold, sub_pct, sub_threshold_policy.max_additional_chunks,
                )

        return reranked, k_final

    def _stage_context_expand(
        self,
        final: list[dict[str, Any]],
        context_expansion_enabled: bool,
    ) -> list[RetrievalResult]:
        """Stage 7: Expand to parent text and build RetrievalResult list."""
        parent_cache: dict[str, str] = {}
        if context_expansion_enabled:
            parent_ids = {
                metadata.get("parent_id")
                for metadata in (
                    item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    for item in final
                )
                if isinstance(metadata.get("parent_id"), str) and metadata.get("parent_id")
            }
            parent_cache = self._storage.get_parent_texts(parent_ids)

        results: list[RetrievalResult] = []
        for item in final:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            parent_id = metadata.get("parent_id")
            parent_text = item.get("parent_text")
            if parent_text is None and isinstance(parent_id, str):
                parent_text = parent_cache.get(parent_id)
            results.append(RetrievalResult(
                child_id=item["id"], text=item.get("text", ""), metadata=metadata,
                score=float(item.get("rerank_score", item.get("score", 0.0))), parent_text=parent_text,
            ))
        return results

    @staticmethod
    def _build_retrieval_documents_from_items(
        items: list[dict[str, Any]],
        *,
        limit: int = 8,
        max_content_chars: int = 400,
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for item in items:
            if len(documents) >= max(1, limit):
                break
            documents.append(
                format_openinference_document(
                    document_id=item.get("id", "unknown"),
                    content=item.get("text") or "",
                    score=item.get("rerank_score", item.get("score", 0.0)),
                    metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    max_content_chars=max_content_chars,
                )
            )
        return documents

    @staticmethod
    def _build_retrieval_documents_from_results(
        results: list[RetrievalResult],
        *,
        limit: int = 8,
        max_content_chars: int = 400,
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for result in results:
            if len(documents) >= max(1, limit):
                break
            documents.append(
                format_openinference_document(
                    document_id=result.child_id,
                    content=result.text or "",
                    score=result.score,
                    metadata=result.metadata,
                    max_content_chars=max_content_chars,
                )
            )
        return documents

    def search(
        self,
        query: str,
        *,
        top_k_fused: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        top_k_final: Optional[int] = None,
        source_id: Optional[str] = None,
        collect_metrics: bool = True,
        params: Optional[ResolvedRetrievalParams] = None,
        retrieval_budget: Optional[int] = None,
        embedding_query: Optional[str] = None,
        bm25_query: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Execute hybrid search with timing and metrics collection.

        Pipeline stages:
        1. Hybrid search (vector ANN + FTS BM25 w/ RRF)
        2. Rerank (Jina v3)
        3. Dedup by parent
        4. Threshold filter (adaptive)
        5. Budget-aware expansion
        6. Final dedup
        7. Context expansion to parent text
        """
        cfg = self._config

        if params is not None:
            k_fused = params.top_k_fused
            k_rerank = params.top_k_rerank
            k_final = params.top_k_final
            reranker_threshold = params.reranker_threshold
            reranker_min_docs = params.reranker_min_docs
        else:
            k_fused = top_k_fused or (cfg.top_k_fused if cfg else 50)
            k_rerank = top_k_rerank or (cfg.top_k_rerank if cfg else 20)
            k_final = top_k_final or (cfg.top_k_final if cfg else 5)
            reranker_threshold = float(cfg.reranker_threshold) if cfg else 0.05
            reranker_min_docs = cfg.reranker_min_docs if cfg else 3

        reranker_enabled = bool(cfg.reranker_enabled) if cfg else self._reranker is not None
        context_expansion_enabled = bool(cfg.context_expansion_enabled) if cfg else True
        if params is not None:
            _max_children = params.max_children_per_parent
        else:
            _max_children = cfg.max_children_per_parent if cfg else 2

        timing = TimingMetrics()
        total_start = time.perf_counter()

        _embedding_q = embedding_query or query
        _bm25_q = bm25_query or query

        with start_span(
            self._tracer,
            "retrieval.search",
            span_kind=SPAN_KIND_RETRIEVER,
            attributes={
                "input.value": query,
                "input.mime_type": "text/plain",
                "rag.intent": intent,
                "rag.source_id": source_id,
                "rag.query_chars": len(query),
                "rag.embedding_query_chars": len(_embedding_q),
                "rag.bm25_query_chars": len(_bm25_q),
                "rag.top_k_fused": k_fused,
                "rag.top_k_rerank": k_rerank,
                "rag.top_k_final": k_final,
                "rag.reranker_enabled": reranker_enabled,
                "rag.context_expansion_enabled": context_expansion_enabled,
            },
        ) as search_span:
            # Stage 1: Hybrid search
            with start_span(
                self._tracer,
                "retrieval.stage.hybrid_search",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                fused, _ = self._stage_hybrid_search(
                    query,
                    _embedding_q,
                    _bm25_q,
                    k_fused,
                    source_id,
                    timing,
                )
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "hybrid_search",
                        "rag.results_count": len(fused),
                        "rag.query_embedding_ms": timing.query_embedding_ms,
                        "rag.hybrid_search_ms": timing.hybrid_search_ms,
                    },
                )

            # Stage 2: Rerank
            with start_span(
                self._tracer,
                "retrieval.stage.rerank",
                span_kind=SPAN_KIND_RERANKER,
            ) as stage_span:
                # Build input documents for RERANKER span before reranking
                input_docs_for_span = self._build_retrieval_documents_from_items(
                    fused[:k_rerank], limit=k_rerank, max_content_chars=220,
                )
                reranked, raw_scores = self._stage_rerank(
                    query,
                    fused,
                    k_rerank,
                    reranker_enabled,
                    timing,
                )
                # Build output documents for RERANKER span after reranking
                output_docs_for_span = self._build_retrieval_documents_from_items(
                    reranked, limit=k_rerank, max_content_chars=220,
                )
                rerank_min = min(raw_scores) if raw_scores else 0.0
                rerank_max = max(raw_scores) if raw_scores else 0.0
                rerank_mean = (sum(raw_scores) / len(raw_scores)) if raw_scores else 0.0
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "rerank",
                        "rag.rerank_ms": timing.rerank_ms,
                        "rag.items_in": len(fused),
                        "rag.items_out": len(reranked),
                        "rag.rerank.score_min": rerank_min,
                        "rag.rerank.score_max": rerank_max,
                        "rag.rerank.score_mean": rerank_mean,
                    },
                )
                set_reranker_documents(
                    stage_span,
                    input_documents=input_docs_for_span,
                    output_documents=output_docs_for_span,
                    query=query,
                    top_k=k_rerank,
                )
            all_reranked = list(reranked)

            # Stage 3: Dedup by parent
            with start_span(
                self._tracer,
                "retrieval.stage.dedup_initial",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                t0 = time.perf_counter()
                reranked, dedup_metrics = self._deduplicate_by_parent(
                    reranked,
                    top_k=len(reranked),
                    max_children_per_parent=_max_children,
                )
                timing.dedup_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "Parent dedup (post-rerank): %d -> %d (%.1f%% reduction)",
                    dedup_metrics.children_before_dedup,
                    dedup_metrics.children_after_dedup,
                    dedup_metrics.reduction_pct,
                )
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "dedup_initial",
                        "rag.dedup_ms": timing.dedup_ms,
                        "rag.children_before": dedup_metrics.children_before_dedup,
                        "rag.children_after": dedup_metrics.children_after_dedup,
                        "rag.parents_deduplicated": dedup_metrics.parents_deduplicated,
                    },
                )

            # Stage 4: Threshold filter
            with start_span(
                self._tracer,
                "retrieval.stage.threshold_filter",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                reranked, above_threshold_results, threshold, threshold_metrics = self._stage_threshold_filter(
                    reranked,
                    reranker_enabled,
                    reranker_threshold,
                    reranker_min_docs,
                )
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "threshold_filter",
                        "rag.threshold": threshold,
                        "rag.items_before": threshold_metrics.items_before_threshold,
                        "rag.items_after": threshold_metrics.items_after_threshold,
                        "rag.safety_net_triggered": threshold_metrics.safety_net_triggered,
                        "input.value": f"{threshold_metrics.items_before_threshold} candidates before thresholding",
                        "input.mime_type": "text/plain",
                        "output.value": (
                            f"{threshold_metrics.items_after_threshold} candidates after threshold={threshold:.6f}"
                        ),
                        "output.mime_type": "text/plain",
                    },
                )
                set_retrieval_documents(
                    stage_span,
                    self._build_retrieval_documents_from_items(reranked),
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

            # Stage 5: Budget expansion
            budget = retrieval_budget or (cfg.retrieval_budget if cfg else 0)
            with start_span(
                self._tracer,
                "retrieval.stage.budget_expand",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                before_expand_count = len(reranked)
                before_k_final = k_final
                reranked, k_final = self._stage_budget_expand(
                    query,
                    reranked,
                    all_reranked,
                    above_threshold_results,
                    k_final,
                    budget,
                    threshold,
                    intent,
                )
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "budget_expand",
                        "rag.retrieval_budget": budget,
                        "rag.threshold": threshold,
                        "rag.items_before": before_expand_count,
                        "rag.items_after": len(reranked),
                        "rag.k_final_before": before_k_final,
                        "rag.k_final_after": k_final,
                    },
                )

            # Stage 6: Final dedup
            with start_span(
                self._tracer,
                "retrieval.stage.final_dedup",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                final: list[dict[str, Any]] = []
                parent_counts: dict[str, int] = {}
                seen_children: set[str] = set()
                for item in reranked:
                    child_id = item.get("id")
                    if isinstance(child_id, str):
                        if child_id in seen_children:
                            continue
                        seen_children.add(child_id)
                    metadata = item.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    parent_id = metadata.get("parent_id")
                    if isinstance(parent_id, str):
                        count = parent_counts.get(parent_id, 0)
                        if count >= _max_children:
                            continue
                        parent_counts[parent_id] = count + 1
                    final.append(item)
                    if len(final) >= k_final:
                        break
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "final_dedup",
                        "rag.items_before": len(reranked),
                        "rag.items_after": len(final),
                    },
                )

            # Stage 7: Context expansion
            with start_span(
                self._tracer,
                "retrieval.stage.context_expand",
                span_kind=SPAN_KIND_RETRIEVER,
            ) as stage_span:
                results = self._stage_context_expand(final, context_expansion_enabled)
                unique_sources = {
                    result.metadata.get("source_id")
                    for result in results
                    if isinstance(result.metadata, dict) and result.metadata.get("source_id")
                }
                set_span_attributes(
                    stage_span,
                    {
                        "rag.stage": "context_expand",
                        "rag.results_count": len(results),
                        "rag.sources_count": len(unique_sources),
                        "output.mime_type": "text/plain",
                        "output.value": f"{len(results)} retrieved passages",
                    },
                )

            timing.total_ms = (time.perf_counter() - total_start) * 1000

            if collect_metrics and results:
                metrics = RetrievalMetrics(
                    timing=timing,
                    reranker=reranker_metrics,
                    deduplication=dedup_metrics,
                    threshold=threshold_metrics,
                    query=query,
                    mode=cfg.mode if cfg else "unknown",
                )
                results[0] = RetrievalResult(
                    child_id=results[0].child_id,
                    text=results[0].text,
                    metadata=results[0].metadata,
                    score=results[0].score,
                    parent_text=results[0].parent_text,
                    metrics=metrics,
                )

            top_results_preview = []
            top_source_ids: list[str] = []
            for rank, result in enumerate(results[:5], start=1):
                metadata = result.metadata if isinstance(result.metadata, dict) else {}
                source_value = metadata.get("source_id")
                if isinstance(source_value, str) and source_value:
                    top_source_ids.append(source_value)
                top_results_preview.append(
                    {
                        "rank": rank,
                        "source_id": source_value,
                        "display_page": metadata.get("display_page"),
                        "page_number": metadata.get("page_number"),
                        "chunk_id": result.child_id,
                        "score": round(float(result.score), 6),
                    }
                )

            set_span_attributes(
                search_span,
                {
                    "rag.total_ms": timing.total_ms,
                    "rag.query_embedding_ms": timing.query_embedding_ms,
                    "rag.hybrid_search_ms": timing.hybrid_search_ms,
                    "rag.rerank_ms": timing.rerank_ms,
                    "rag.dedup_ms": timing.dedup_ms,
                    "rag.results_count": len(results),
                    "rag.threshold": threshold,
                    "rag.reranker.items_reranked": reranker_metrics.items_reranked,
                    "rag.reranker.score_min": reranker_metrics.score_min,
                    "rag.reranker.score_max": reranker_metrics.score_max,
                    "rag.threshold.items_after": threshold_metrics.items_after_threshold,
                    "rag.threshold.safety_net": threshold_metrics.safety_net_triggered,
                    "rag.top_source_ids": sorted(set(top_source_ids))[:10],
                    "rag.top_results": top_results_preview,
                    "output.mime_type": "text/plain",
                    "output.value": f"{len(results)} retrieval results",
                },
            )
            set_retrieval_documents(
                search_span,
                self._build_retrieval_documents_from_results(results),
            )

            return results
