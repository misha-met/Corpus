"""Corpus-level analytics: topics, entities, timeline, and relationship data.

All computation is synchronous and CPU-bound.  Callers must run this in a
thread pool (``asyncio.to_thread``) to avoid blocking the event loop.

Graceful degradation
~~~~~~~~~~~~~~~~~~~~
- spaCy: if ``en_core_web_sm`` is not installed, NER returns empty results
  and ``ner_available`` is False in the response.
- dateparser: if not installed, timeline returns empty results and
  ``timeline_available`` is False.
- scikit-learn: required for TF-IDF topics; if not installed, topic clusters
  return empty results.

Caching
~~~~~~~
Results are cached to ``data/analytics_cache/corpus_analytics.json``.
The cache is invalidated by hashing the sorted source IDs + count.
Pass ``force_recompute=True`` to bypass the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
_DATEPARSER_AVAILABLE = False
_SKLEARN_AVAILABLE = False

try:
    import spacy  # noqa: F401
    try:
        _nlp = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        _nlp = None
        logger.info("spaCy model 'en_core_web_sm' not found; NER disabled")
except ImportError:
    _nlp = None
    logger.info("spaCy not installed; NER disabled")

try:
    import dateparser.search  # noqa: F401
    _DATEPARSER_AVAILABLE = True
except ImportError:
    logger.info("dateparser not installed; timeline disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.info("scikit-learn not installed; topic clustering disabled")

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("data/analytics_cache")
_CACHE_FILE = _CACHE_DIR / "corpus_analytics.json"


def _cache_key(source_ids: list[str]) -> str:
    payload = ",".join(sorted(source_ids)) + f"|n={len(source_ids)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_cache(source_ids: list[str]) -> Optional[dict[str, Any]]:
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if data.get("_cache_key") == _cache_key(source_ids):
            return data
    except Exception as exc:
        logger.debug("Cache read failed: %s", exc)
    return None


def _write_cache(data: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write analytics cache: %s", exc)


# ---------------------------------------------------------------------------
# Corpus overview
# ---------------------------------------------------------------------------

def _compute_overview(storage) -> dict[str, Any]:
    """Compute basic corpus statistics from LanceDB tables."""
    source_ids = storage.list_source_ids()
    source_count = len(source_ids)

    child_count = 0
    parent_count = 0
    total_tokens = 0

    if storage._table is not None:
        try:
            rows = storage._table.to_arrow().to_pylist()
            child_count = len(rows)
            for row in rows:
                text = row.get("text", "") or ""
                total_tokens += int(len(text.split()) * 1.35)
        except Exception as exc:
            logger.warning("Could not count child chunks: %s", exc)

    if storage._parents is not None:
        try:
            parent_count = storage._parents.count_rows()
        except Exception as exc:
            logger.warning("Could not count parent chunks: %s", exc)

    avg_chunks_per_doc = round(child_count / source_count, 1) if source_count > 0 else 0.0

    return {
        "source_count": source_count,
        "child_chunk_count": child_count,
        "parent_chunk_count": parent_count,
        "estimated_tokens": total_tokens,
        "avg_chunks_per_doc": avg_chunks_per_doc,
        "source_ids": source_ids,
    }


# ---------------------------------------------------------------------------
# TF-IDF topic clustering
# ---------------------------------------------------------------------------

def _compute_topics(storage, source_ids: list[str]) -> list[dict[str, Any]]:
    """Cluster corpus by TF-IDF + truncated SVD + agglomerative/k-means."""
    if not _SKLEARN_AVAILABLE or not source_ids:
        return []

    # Group child chunk texts by source
    source_texts: dict[str, list[str]] = {sid: [] for sid in source_ids}
    if storage._table is not None:
        try:
            rows = storage._table.to_arrow().to_pylist()
            for row in rows:
                sid = row.get("source_id", "")
                text = row.get("text", "") or ""
                if sid in source_texts and text:
                    source_texts[sid].append(text)
        except Exception as exc:
            logger.warning("TF-IDF: failed to load child chunks: %s", exc)
            return []

    # Build per-source document (concatenate chunks)
    docs = []
    doc_source_ids = []
    for sid in source_ids:
        combined = " ".join(source_texts[sid])
        if combined.strip():
            docs.append(combined)
            doc_source_ids.append(sid)

    if len(docs) < 2:
        # Not enough documents to cluster
        if len(docs) == 1:
            return [{
                "cluster_id": 0,
                "label": "All Documents",
                "keywords": [],
                "source_ids": doc_source_ids,
                "size": len(doc_source_ids),
            }]
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            max_features=5000,
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        n_docs = len(docs)
        n_components = min(20, n_docs - 1) if n_docs > 1 else 1
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)
        reduced_norm = normalize(reduced)

        # Auto-select k via silhouette score
        k_range = range(2, min(10, n_docs // 2 + 1) + 1) if n_docs >= 4 else [min(2, n_docs)]
        best_k = 2
        best_score = -1.0
        for k in k_range:
            if k >= n_docs:
                continue
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(reduced_norm)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(reduced_norm, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                pass

        # Final clustering
        if n_docs < 50:
            clusterer = AgglomerativeClustering(n_clusters=best_k)
            labels = clusterer.fit_predict(reduced_norm)
        else:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km.fit_predict(reduced_norm)

        # Extract top keywords per cluster from TF-IDF centroid
        # Compute cluster centroids in original TF-IDF space
        clusters: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(i)

        result = []
        for cluster_id, indices in sorted(clusters.items()):
            # Mean TF-IDF vector for the cluster
            cluster_matrix = tfidf_matrix[indices]
            centroid = np.asarray(cluster_matrix.mean(axis=0)).flatten()
            top_indices = centroid.argsort()[-8:][::-1]
            keywords = [feature_names[i] for i in top_indices if centroid[i] > 0]

            cluster_source_ids = [doc_source_ids[i] for i in indices]
            result.append({
                "cluster_id": cluster_id,
                "label": keywords[0] if keywords else f"Cluster {cluster_id}",
                "keywords": keywords,
                "source_ids": cluster_source_ids,
                "size": len(cluster_source_ids),
            })

        return result

    except Exception as exc:
        logger.warning("Topic clustering failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Named entity recognition
# ---------------------------------------------------------------------------

_ENTITY_NORMALIZE_RE = re.compile(r"\s+")
_ENTITY_PERIOD_RE = re.compile(r"\.+$")


def _normalize_entity(text: str) -> str:
    text = _ENTITY_NORMALIZE_RE.sub(" ", text).strip()
    text = _ENTITY_PERIOD_RE.sub("", text)
    return text.title()


def _compute_entities(storage, source_ids: list[str]) -> list[dict[str, Any]]:
    """Extract named entities via spaCy from source summaries + first 50 parents."""
    if not _SPACY_AVAILABLE or _nlp is None or not source_ids:
        return []

    texts_to_process: list[str] = []

    # Add summaries
    if storage._summaries is not None:
        try:
            rows = storage._summaries.to_arrow().to_pylist()
            for row in rows:
                summary = row.get("summary", "") or ""
                if summary.strip():
                    texts_to_process.append(summary)
        except Exception as exc:
            logger.warning("NER: failed to load summaries: %s", exc)

    # Add first 50 parent chunks per source
    if storage._parents is not None:
        try:
            for sid in source_ids:
                rows = (
                    storage._parents
                    .search()
                    .where(f"source_id = '{storage._escape_sql_literal(sid)}'", prefilter=True)
                    .select(["text"])
                    .limit(50)
                    .to_list()
                )
                for row in rows:
                    text = row.get("text", "") or ""
                    if text.strip():
                        texts_to_process.append(text)
        except Exception as exc:
            logger.warning("NER: failed to load parent chunks: %s", exc)

    if not texts_to_process:
        return []

    # Run NER
    entity_counts: dict[tuple[str, str], int] = {}
    try:
        for doc in _nlp.pipe(texts_to_process, batch_size=32, disable=["parser", "lemmatizer"]):
            for ent in doc.ents:
                if ent.label_ not in ("PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "WORK_OF_ART", "FAC"):
                    continue
                normalized = _normalize_entity(ent.text)
                if len(normalized) < 2:
                    continue
                key = (normalized, ent.label_)
                entity_counts[key] = entity_counts.get(key, 0) + 1
    except Exception as exc:
        logger.warning("NER processing failed: %s", exc)
        return []

    # Return top 50 by count, frequency ≥ 1
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    return [
        {"text": text, "type": etype, "count": count}
        for (text, etype), count in sorted_entities
    ]


# ---------------------------------------------------------------------------
# Temporal distribution
# ---------------------------------------------------------------------------

# Regex patterns for date extraction when dateparser is not available
_CENTURY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)\s+century\b", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_DECADE_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])s\b", re.IGNORECASE)


def _text_to_year_approx(text: str) -> list[int]:
    """Extract approximate years from text using regex fallbacks."""
    years: list[int] = []
    for m in _CENTURY_RE.finditer(text):
        century = int(m.group(1))
        years.append((century - 1) * 100 + 50)  # midpoint of century
    for m in _DECADE_RE.finditer(text):
        years.append(int(m.group(1)) + 5)  # midpoint of decade
    for m in _YEAR_RE.finditer(text):
        years.append(int(m.group(1)))
    return years


def _compute_timeline(storage, source_ids: list[str]) -> list[dict[str, Any]]:
    """Build temporal distribution buckets from corpus text."""
    if not source_ids:
        return []

    year_source_map: dict[int, set[str]] = {}

    texts_to_scan: list[tuple[str, str]] = []  # (text, source_id)

    # Use summaries + first 20 parent chunks for speed
    if storage._summaries is not None:
        try:
            rows = storage._summaries.to_arrow().to_pylist()
            for row in rows:
                sid = row.get("source_id", "")
                summary = row.get("summary", "") or ""
                if summary.strip() and sid:
                    texts_to_scan.append((summary, sid))
        except Exception as exc:
            logger.debug("Timeline: failed to load summaries: %s", exc)

    if storage._parents is not None:
        try:
            for sid in source_ids:
                rows = (
                    storage._parents
                    .search()
                    .where(f"source_id = '{storage._escape_sql_literal(sid)}'", prefilter=True)
                    .select(["text"])
                    .limit(20)
                    .to_list()
                )
                for row in rows:
                    text = row.get("text", "") or ""
                    if text.strip():
                        texts_to_scan.append((text, sid))
        except Exception as exc:
            logger.debug("Timeline: failed to load parent chunks: %s", exc)

    for text, sid in texts_to_scan:
        if _DATEPARSER_AVAILABLE:
            try:
                import dateparser.search
                results = dateparser.search.search_dates(
                    text[:2000],  # limit text length for speed
                    settings={"STRICT_PARSING": True, "PREFER_DAY_OF_MONTH": "first"},
                )
                if results:
                    for _, dt in results:
                        year = dt.year
                        if 900 <= year <= 2030:
                            year_source_map.setdefault(year, set()).add(sid)
            except Exception:
                pass
        # Always run regex as supplement
        for year in _text_to_year_approx(text):
            if 900 <= year <= 2030:
                year_source_map.setdefault(year, set()).add(sid)

    if not year_source_map:
        return []

    # Bucket into decades
    decade_counts: dict[int, dict[str, Any]] = {}
    for year, sids in year_source_map.items():
        decade = (year // 10) * 10
        if decade not in decade_counts:
            decade_counts[decade] = {"count": 0, "sources": set()}
        decade_counts[decade]["count"] += len(sids)
        decade_counts[decade]["sources"].update(sids)

    buckets = []
    for decade in sorted(decade_counts):
        entry = decade_counts[decade]
        buckets.append({
            "period_start": decade,
            "period_end": decade + 9,
            "label": f"{decade}s",
            "count": entry["count"],
            "sources": sorted(entry["sources"]),
        })

    return buckets


# ---------------------------------------------------------------------------
# Source relationship graph
# ---------------------------------------------------------------------------

def _compute_relationships(
    storage,
    source_ids: list[str],
    topics: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute source relationship graph with embedding, entity, and temporal edges.

    Memory-efficient: fetches vectors one source at a time, computes the mean
    centroid, then discards the raw vectors.  Only the centroid matrix
    (n_sources × embedding_dim) is held in memory when calling cosine_similarity.
    """
    if not _SKLEARN_AVAILABLE or len(source_ids) < 2:
        # Build minimal nodes even for a single-source corpus
        nodes = [{"id": sid, "label": sid, "size": 0, "dominant_topic": None, "summary": None} for sid in source_ids]
        return {"nodes": nodes, "edges": []}

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # ── Node metadata ───────────────────────────────────────────────────────

    # Dominant topic per source (from already-computed topic clusters)
    topic_map: dict[str, Optional[int]] = {sid: None for sid in source_ids}
    for topic in topics:
        for sid in topic.get("source_ids", []):
            if sid in topic_map and topic_map[sid] is None:
                topic_map[sid] = topic.get("cluster_id")

    # First-200-char summary per source
    summary_map: dict[str, str] = {}
    if storage._summaries is not None:
        try:
            rows = storage._summaries.to_arrow().to_pylist()
            for row in rows:
                sid = row.get("source_id", "")
                text = (row.get("summary", "") or "")[:200]
                if sid in topic_map and text:
                    summary_map[sid] = text
        except Exception as exc:
            logger.warning("Relationships: summaries failed: %s", exc)

    # ── Batch centroid computation (memory-efficient) ────────────────────────

    size_map: dict[str, int] = {sid: 0 for sid in source_ids}
    centroids: dict[str, Any] = {}  # sid → np.ndarray

    if storage._table is not None:
        for sid in source_ids:
            try:
                escaped = storage._escape_sql_literal(sid)
                rows = (
                    storage._table
                    .search()
                    .where(f"source_id = '{escaped}'", prefilter=True)
                    .select(["vector"])
                    .limit(10_000)
                    .to_list()
                )
                if not rows:
                    continue
                size_map[sid] = len(rows)
                vecs = np.array([r["vector"] for r in rows], dtype=np.float32)
                centroids[sid] = vecs.mean(axis=0)
                del vecs  # free memory immediately
            except Exception as exc:
                logger.warning("Relationships: centroid for %s failed: %s", sid, exc)

    # ── Pairwise embedding similarity ────────────────────────────────────────

    sim_pairs: dict[tuple[str, str], float] = {}
    valid_sids = [s for s in source_ids if s in centroids]
    if len(valid_sids) >= 2:
        try:
            centroid_matrix = np.array([centroids[s] for s in valid_sids], dtype=np.float32)
            sim_matrix = cosine_similarity(centroid_matrix)  # shape (n, n)
            for i, si in enumerate(valid_sids):
                row = sim_matrix[i].copy()
                row[i] = -1.0  # exclude self
                top_j_indices = row.argsort()[-8:][::-1]
                for j in top_j_indices:
                    if j == i:
                        continue  # guard: when n < 8, i appears in results
                    score = float(sim_matrix[i, j])
                    if score < 0.3:
                        continue
                    # canonical key: lower index first to avoid duplicates
                    a, b = (si, valid_sids[j]) if i < j else (valid_sids[j], si)
                    if (a, b) not in sim_pairs or sim_pairs[(a, b)] < score:
                        sim_pairs[(a, b)] = score
        except Exception as exc:
            logger.warning("Relationships: cosine similarity failed: %s", exc)

    # ── Per-source entity sets ────────────────────────────────────────────────
    # Re-extract compact entity sets per source (uses parent chunks, max 20 each)

    entity_sets: dict[str, set[str]] = {sid: set() for sid in source_ids}
    if _SPACY_AVAILABLE and _nlp is not None and storage._parents is not None:
        try:
            for sid in source_ids:
                escaped = storage._escape_sql_literal(sid)
                rows = (
                    storage._parents
                    .search()
                    .where(f"source_id = '{escaped}'", prefilter=True)
                    .select(["text"])
                    .limit(20)
                    .to_list()
                )
                texts = [r.get("text", "") or "" for r in rows if r.get("text")]
                for doc in _nlp.pipe(texts, batch_size=16, disable=["parser", "lemmatizer"]):
                    for ent in doc.ents:
                        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP"):
                            entity_sets[sid].add(_normalize_entity(ent.text))
        except Exception as exc:
            logger.warning("Relationships: entity sets failed: %s", exc)

    # ── Jaccard entity similarity ─────────────────────────────────────────────

    entity_pairs: dict[tuple[str, str], float] = {}
    sid_list = source_ids
    for i, si in enumerate(sid_list):
        for j in range(i + 1, len(sid_list)):
            sj = sid_list[j]
            set_i = entity_sets.get(si, set())
            set_j = entity_sets.get(sj, set())
            if not set_i or not set_j:
                continue
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            if union > 0:
                jaccard = intersection / union
                if jaccard >= 0.15:
                    entity_pairs[(si, sj)] = jaccard

    # ── Temporal overlap ──────────────────────────────────────────────────────
    # Derive per-source year ranges from already-computed timeline buckets

    source_years: dict[str, list[int]] = {sid: [] for sid in source_ids}
    for bucket in timeline:
        mid = (bucket.get("period_start", 0) + bucket.get("period_end", 0)) // 2
        for sid in bucket.get("sources", []):
            if sid in source_years:
                source_years[sid].append(mid)

    year_ranges: dict[str, tuple[int, int]] = {}
    for sid, years in source_years.items():
        if years:
            year_ranges[sid] = (min(years), max(years))

    temporal_pairs: dict[tuple[str, str], float] = {}
    for i, si in enumerate(sid_list):
        for j in range(i + 1, len(sid_list)):
            sj = sid_list[j]
            ri = year_ranges.get(si)
            rj = year_ranges.get(sj)
            if ri is None or rj is None:
                continue
            overlap_start = max(ri[0], rj[0])
            overlap_end = min(ri[1], rj[1])
            if overlap_end > overlap_start:
                total = max(ri[1], rj[1]) - min(ri[0], rj[0])
                if total > 0:
                    ratio = (overlap_end - overlap_start) / total
                    if ratio >= 0.1:
                        temporal_pairs[(si, sj)] = ratio

    # ── Merge edges ───────────────────────────────────────────────────────────

    all_pairs: set[tuple[str, str]] = set(sim_pairs) | set(entity_pairs) | set(temporal_pairs)
    edges: list[dict[str, Any]] = []
    for (si, sj) in all_pairs:
        types: list[str] = []
        weights: dict[str, float] = {}
        if (si, sj) in sim_pairs:
            types.append("similarity")
            weights["similarity"] = sim_pairs[(si, sj)]
        if (si, sj) in entity_pairs:
            types.append("entities")
            weights["entities"] = entity_pairs[(si, sj)]
        if (si, sj) in temporal_pairs:
            types.append("temporal")
            weights["temporal"] = temporal_pairs[(si, sj)]
        combined = (
            0.5 * weights.get("similarity", 0.0)
            + 0.3 * weights.get("entities", 0.0)
            + 0.2 * weights.get("temporal", 0.0)
        )
        edges.append({
            "source": si,
            "target": sj,
            "types": types,
            "weights": weights,
            "combined_weight": round(combined, 4),
        })

    nodes = [
        {
            "id": sid,
            "label": sid,
            "size": size_map.get(sid, 0),
            "dominant_topic": topic_map.get(sid),
            "summary": summary_map.get(sid),
        }
        for sid in source_ids
    ]

    logger.info(
        "Relationships: %d nodes, %d edges (%d sim, %d entity, %d temporal)",
        len(nodes),
        len(edges),
        len(sim_pairs),
        len(entity_pairs),
        len(temporal_pairs),
    )
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_corpus_analytics(
    storage,
    *,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Compute (or return cached) full corpus analytics.

    Parameters
    ----------
    storage:
        A ``StorageEngine`` instance.
    force_recompute:
        If True, ignore the cache and recompute from scratch.

    Returns
    -------
    dict with keys: overview, topics, entities, timeline,
    ner_available, timeline_available
    """
    source_ids = storage.list_source_ids()

    if not force_recompute:
        cached = _read_cache(source_ids)
        if cached is not None:
            logger.info("Analytics: returning cached result (key=%s)", cached.get("_cache_key"))
            return cached

    logger.info(
        "Analytics: computing for %d sources (spacy=%s, dateparser=%s, sklearn=%s)",
        len(source_ids),
        _SPACY_AVAILABLE,
        _DATEPARSER_AVAILABLE,
        _SKLEARN_AVAILABLE,
    )

    overview = _compute_overview(storage)
    topics = _compute_topics(storage, source_ids) if _SKLEARN_AVAILABLE else []
    entities = _compute_entities(storage, source_ids) if _SPACY_AVAILABLE else []
    timeline = _compute_timeline(storage, source_ids)
    relationships = _compute_relationships(storage, source_ids, topics, timeline)

    result: dict[str, Any] = {
        "_cache_key": _cache_key(source_ids),
        "overview": overview,
        "topics": topics,
        "entities": entities,
        "timeline": timeline,
        "relationships": relationships,
        "ner_available": _SPACY_AVAILABLE,
        "timeline_available": _DATEPARSER_AVAILABLE,
    }

    _write_cache(result)
    logger.info("Analytics: computation complete, cached")
    return result
