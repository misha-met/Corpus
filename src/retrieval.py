from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .storage import StorageEngine


@dataclass(frozen=True)
class RetrievalResult:
    child_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    parent_text: Optional[str]


class RetrievalEngine:
    def __init__(
        self,
        *,
        storage: StorageEngine,
        embedding_model: Any,
        reranker: Optional[Any] = None,
    ) -> None:
        self._storage = storage
        self._embedding_model = embedding_model
        self._reranker = reranker

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        try:
            embeddings = self._embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("Embedding model encode failed.") from exc

        response = self._storage.query_children(embeddings=embeddings, top_k=top_k)

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

    def _sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
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
        )[:top_k]

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            child_id = self._storage.bm25_ids[idx]
            results.append(
                {
                    "id": child_id,
                    "score": float(score),
                    "rank": rank,
                }
            )
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

    def _rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._reranker is None or not items:
            return items

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

        reranked = []
        for item, score in zip(items, scores):
            reranked.append({**item, "rerank_score": float(score)})
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked

    def search(
        self,
        query: str,
        *,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        top_k_fused: int = 20,
        top_k_final: int = 5,
    ) -> list[RetrievalResult]:
        dense = self._dense_search(query, top_k_dense)
        sparse = self._sparse_search(query, top_k_sparse)

        fused = self._rrf_fuse(dense, sparse)[:top_k_fused]

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

        parent_cache: dict[str, str] = {}
        for item in fused:
            metadata = item.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if isinstance(parent_id, str):
                if parent_id not in parent_cache:
                    parent_text = self._storage.get_parent_text(parent_id)
                    if parent_text:
                        parent_cache[parent_id] = parent_text
                if parent_id in parent_cache:
                    item["rerank_text"] = parent_cache[parent_id]

        reranked = self._rerank(query, fused)
        final = reranked[:top_k_final]

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

        return results
