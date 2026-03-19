"""LanceDB storage engine for chunks, parents, and source summaries.

Architecture
~~~~~~~~~~~~
- Four LanceDB tables: ``child_chunks`` (dense vectors + FTS text index),
  ``parent_chunks`` (text only, no vectors), ``source_summaries`` (source
  metadata and file paths), and ``geo_mentions`` (ingest-time geocoded mentions).
- FTS index rebuild is controlled by a policy (``immediate``, ``deferred``,
  ``batch``) to trade ingest throughput against query-time freshness.  The
  ``_fts_dirty`` flag tracks whether a rebuild is pending so hybrid_search
  can trigger one lazily before the first query after a write.
- SQL escaping is done manually via ``_escape_sql_literal()`` rather than
  parameterised queries because the LanceDB Python API does not expose
  parameterised WHERE clauses at this version.
- Single-process, single-user design — no connection pooling or row-level
  locking.  Concurrent writes from multiple processes would corrupt the index.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import lancedb
import pyarrow as pa

from .models import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    lance_dir: Path
    lance_table: str = "child_chunks"
    fts_rebuild_policy: str = "immediate"
    fts_rebuild_batch_size: int = 0


class StorageEngine:
    """Unified storage engine backed entirely by LanceDB."""

    _PARENTS_TABLE = "parent_chunks"
    _SUMMARIES_TABLE = "source_summaries"
    _GEO_MENTIONS_TABLE = "geo_mentions"
    _PERSON_MENTIONS_TABLE = "person_mentions"
    _MAX_IN_CLAUSE_VALUES = 256
    _GEO_MENTIONS_SCHEMA = pa.schema([
        pa.field("id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("place_name", pa.string()),
        pa.field("matched_input", pa.string()),
        pa.field("matched_on", pa.string()),
        pa.field("geonameid", pa.int64()),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("method", pa.string()),
        pa.field("raw_score", pa.float64()),
        pa.field("is_ambiguous", pa.bool_()),
        pa.field("candidate_count", pa.int32()),
        pa.field("margin_score", pa.float64()),
        pa.field("entity_type", pa.string()),
        pa.field("ner_score", pa.float64()),
        pa.field("geocoder_version", pa.string()),
        pa.field("geocoded_at", pa.float64()),
    ])
    _PERSON_MENTIONS_SCHEMA = pa.schema([
        pa.field("id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("raw_name", pa.string()),
        pa.field("canonical_name", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("method", pa.string()),
        pa.field("role_hint", pa.string()),
        pa.field("context_snippet", pa.string()),
    ])

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        if config.fts_rebuild_policy not in {"immediate", "deferred", "batch"}:
            raise ValueError("fts_rebuild_policy must be one of: immediate, deferred, batch")
        if config.fts_rebuild_batch_size < 0:
            raise ValueError("fts_rebuild_batch_size must be >= 0")
        config.lance_dir.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(config.lance_dir))
        self._table_name = config.lance_table

        # --- child chunks (vectors + FTS) ---
        self._table: Optional[lancedb.table.Table] = None
        try:
            self._table = self._db.open_table(self._table_name)
            self._migrate_children_schema()
            logger.info("Opened LanceDB table '%s'", self._table_name)
        except ValueError:
            logger.info(
                "LanceDB table '%s' not found; will create on first ingest",
                self._table_name,
            )
        self._fts_dirty = False
        self._pending_fts_rows = 0

        # --- parent chunks (no vectors) ---
        self._parents: Optional[lancedb.table.Table] = None
        try:
            self._parents = self._db.open_table(self._PARENTS_TABLE)
            self._migrate_parents_schema()
        except ValueError:
            pass  # Table does not exist yet

        # --- source summaries (no vectors) ---
        self._summaries: Optional[lancedb.table.Table] = None
        try:
            self._summaries = self._db.open_table(self._SUMMARIES_TABLE)
            self._migrate_summaries_schema()
        except ValueError:
            pass  # Table does not exist yet

        # --- geo mentions (no vectors) ---
        self._geo_mentions: Optional[lancedb.table.Table] = None
        self._geo_mentions_indexes_ready = False
        try:
            self._geo_mentions = self._db.open_table(self._GEO_MENTIONS_TABLE)
            self._migrate_geo_mentions_schema()
            self._ensure_geo_mentions_indexes()
        except ValueError:
            pass  # Table does not exist yet

        # --- person mentions (no vectors) ---
        self._person_mentions: Optional[lancedb.table.Table] = None
        self._person_mentions_indexes_ready = False
        try:
            self._person_mentions = self._db.open_table(self._PERSON_MENTIONS_TABLE)
            self._ensure_person_mentions_indexes()
        except ValueError:
            pass  # Table does not exist yet

    def close(self) -> None:
        pass  # LanceDB connections do not require explicit close

    # ------------------------------------------------------------------
    # Schema migrations
    # ------------------------------------------------------------------

    _SUMMARIES_V2_COLUMNS = ("source_path", "snapshot_path", "citation_reference")
    _SUMMARIES_V3_COLUMNS = ("page_offset",)  # int32; absence treated as 1 at read time
    _RANGE_COLUMNS = ("start_page", "end_page")

    def _migrate_children_schema(self) -> None:
        """Add range columns to child chunk table if missing."""
        if self._table is None:
            return
        existing = set(self._table.schema.names)
        missing: list[pa.Field] = [
            pa.field(col, pa.int32())
            for col in self._RANGE_COLUMNS
            if col not in existing
        ]
        if not missing:
            return
        self._table.add_columns(missing)
        self._table = self._db.open_table(self._table_name)
        logger.info(
            "Migrated '%s' table: added columns %s",
            self._table_name,
            [f.name for f in missing],
        )

    def _migrate_parents_schema(self) -> None:
        """Add range columns to parent chunk table if missing."""
        if self._parents is None:
            return
        existing = set(self._parents.schema.names)
        missing: list[pa.Field] = [
            pa.field(col, pa.int32())
            for col in self._RANGE_COLUMNS
            if col not in existing
        ]
        if not missing:
            return
        self._parents.add_columns(missing)
        self._parents = self._db.open_table(self._PARENTS_TABLE)
        logger.info(
            "Migrated '%s' table: added columns %s",
            self._PARENTS_TABLE,
            [f.name for f in missing],
        )

    def _migrate_summaries_schema(self) -> None:
        """Add v2 and v3 columns to the source_summaries table if missing."""
        if self._summaries is None:
            return
        existing = set(self._summaries.schema.names)
        missing: list[pa.Field] = [
            pa.field(col, pa.utf8())
            for col in self._SUMMARIES_V2_COLUMNS
            if col not in existing
        ]
        if "page_offset" not in existing:
            missing.append(pa.field("page_offset", pa.int32()))
        if not missing:
            return
        self._summaries.add_columns(missing)
        # Re-open so the cached schema is up to date
        self._summaries = self._db.open_table(self._SUMMARIES_TABLE)
        logger.info(
            "Migrated '%s' table: added columns %s",
            self._SUMMARIES_TABLE,
            [f.name for f in missing],
        )

    def _migrate_geo_mentions_schema(self) -> None:
        """Ensure geo_mentions includes all diagnostic columns."""
        if self._geo_mentions is None:
            return
        existing = set(self._geo_mentions.schema.names)
        missing = [field for field in self._GEO_MENTIONS_SCHEMA if field.name not in existing]
        if not missing:
            return
        self._geo_mentions.add_columns(missing)
        self._geo_mentions = self._db.open_table(self._GEO_MENTIONS_TABLE)
        self._geo_mentions_indexes_ready = False
        logger.info(
            "Migrated '%s' table: added columns %s",
            self._GEO_MENTIONS_TABLE,
            [f.name for f in missing],
        )

    def _ensure_geo_mentions_indexes(self) -> None:
        """Best-effort scalar indexes for common geo mention filters."""
        if self._geo_mentions is None or self._geo_mentions_indexes_ready:
            return
        for column in ("source_id", "confidence", "geonameid"):
            try:
                self._geo_mentions.create_scalar_index(column)
            except TypeError:
                # Older LanceDB signatures use keyword arguments.
                try:
                    self._geo_mentions.create_scalar_index(column, replace=False)
                except Exception as exc:
                    logger.debug("Skipping geo_mentions index for %s: %s", column, exc)
            except Exception as exc:
                logger.debug("Skipping geo_mentions index for %s: %s", column, exc)
            self._geo_mentions_indexes_ready = True

    def _ensure_person_mentions_indexes(self) -> None:
        """Best-effort scalar indexes for person mention filters."""
        if self._person_mentions is None or self._person_mentions_indexes_ready:
            return
        for column in ("source_id", "canonical_name", "confidence"):
            try:
                self._person_mentions.create_scalar_index(column)
            except TypeError:
                try:
                    self._person_mentions.create_scalar_index(column, replace=False)
                except Exception as exc:
                    logger.debug("Skipping person_mentions index for %s: %s", column, exc)
            except Exception as exc:
                logger.debug("Skipping person_mentions index for %s: %s", column, exc)
        self._person_mentions_indexes_ready = True

    def _ensure_person_mentions_table(self) -> Optional[lancedb.table.Table]:
        if self._person_mentions is not None:
            self._ensure_person_mentions_indexes()
            return self._person_mentions
        try:
            self._person_mentions = self._db.create_table(
                self._PERSON_MENTIONS_TABLE,
                schema=self._PERSON_MENTIONS_SCHEMA,
                exist_ok=True,
            )
            self._person_mentions_indexes_ready = False
            self._ensure_person_mentions_indexes()
            return self._person_mentions
        except Exception as exc:
            logger.error("Failed to create/open '%s': %s", self._PERSON_MENTIONS_TABLE, exc)
            return None

    def _recreate_person_mentions_table(self) -> Optional[lancedb.table.Table]:
        """Drop and recreate person_mentions for explicit schema reset workflows."""
        try:
            self._db.drop_table(self._PERSON_MENTIONS_TABLE)
        except Exception as exc:
            logger.debug("Could not drop '%s' before recreate: %s", self._PERSON_MENTIONS_TABLE, exc)

        self._person_mentions = None
        self._person_mentions_indexes_ready = False
        return self._ensure_person_mentions_table()

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @classmethod
    def _where_eq(cls, column: str, value: str) -> str:
        return f"{column} = '{cls._escape_sql_literal(value)}'"

    @classmethod
    def _where_in(cls, column: str, values: list[str]) -> str:
        if not values:
            return "1 = 0"
        escaped = ", ".join(f"'{cls._escape_sql_literal(v)}'" for v in values)
        return f"{column} IN ({escaped})"

    @classmethod
    def _where_contains_ci(cls, column: str, value: str) -> str:
        """Case-insensitive SQL substring predicate for Lance/DataFusion backends."""
        escaped = cls._escape_sql_literal(value.lower())
        return f"LOWER({column}) LIKE '%{escaped}%'"

    @staticmethod
    def _chunk(values: list[str], size: int) -> Iterable[list[str]]:
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def _mark_fts_dirty(self, added_rows: int) -> None:
        self._fts_dirty = True
        self._pending_fts_rows += max(0, added_rows)

    def _ensure_fts_index(self, *, force_rebuild: bool = False) -> None:
        if self._table is None:
            return
        if force_rebuild or self._fts_dirty:
            try:
                self._table.create_fts_index("text", replace=True)
            except Exception:
                # Keep dirty state so future writes/queries can retry rebuild.
                self._fts_dirty = True
                raise
            else:
                self._fts_dirty = False
                self._pending_fts_rows = 0

    def _maybe_rebuild_fts_after_write(self) -> None:
        policy = self._config.fts_rebuild_policy
        if policy == "immediate":
            self._ensure_fts_index(force_rebuild=True)
            logger.info("Rebuilt FTS index immediately after write")
            return

        if policy == "batch":
            batch_size = self._config.fts_rebuild_batch_size
            if batch_size > 0 and self._pending_fts_rows >= batch_size:
                self._ensure_fts_index(force_rebuild=True)
                logger.info(
                    "Rebuilt FTS index after batch threshold (%d rows)",
                    batch_size,
                )
                return

        logger.debug(
            "Deferred FTS rebuild (policy=%s, pending_rows=%d)",
            self._config.fts_rebuild_policy,
            self._pending_fts_rows,
        )

    def _refresh_fts_if_dirty_for_query(self) -> None:
        if self._fts_dirty:
            self._ensure_fts_index(force_rebuild=True)
            logger.info("Refreshed dirty FTS index before hybrid search")

    def get_fts_status(self) -> dict[str, Any]:
        return {
            "fts_policy": self._config.fts_rebuild_policy,
            "fts_dirty": self._fts_dirty,
            "fts_pending_rows": int(self._pending_fts_rows),
        }

    def get_child_vector_dimension(self) -> Optional[int]:
        """Return child vector dimension if available, otherwise None."""
        if self._table is None:
            return None
        try:
            field = self._table.schema.field("vector")
        except Exception:
            return None

        field_type = field.type
        list_size = getattr(field_type, "list_size", None)
        if isinstance(list_size, int) and list_size > 0:
            return int(list_size)

        value_type = getattr(field_type, "value_type", None)
        nested_list_size = getattr(value_type, "list_size", None)
        if isinstance(nested_list_size, int) and nested_list_size > 0:
            return int(nested_list_size)
        return None

    def reset_all_tables(self) -> None:
        """Drop all managed tables and clear in-memory handles."""
        for table_name in (
            self._table_name,
            self._PARENTS_TABLE,
            self._SUMMARIES_TABLE,
            self._GEO_MENTIONS_TABLE,
            self._PERSON_MENTIONS_TABLE,
        ):
            try:
                self._db.drop_table(table_name)
                logger.warning("Dropped LanceDB table '%s'", table_name)
            except Exception as exc:
                logger.debug("Could not drop table '%s': %s", table_name, exc)

        self._table = None
        self._parents = None
        self._summaries = None
        self._geo_mentions = None
        self._person_mentions = None
        self._geo_mentions_indexes_ready = False
        self._person_mentions_indexes_ready = False
        self._fts_dirty = False
        self._pending_fts_rows = 0

    def add_parents(self, parents: Iterable[ParentChunk]) -> None:
        records = [
            {
                "parent_id": p.id,
                "source_id": p.metadata.source_id,
                "page_number": p.metadata.page_number or 0,
                "start_page": p.metadata.start_page,
                "end_page": p.metadata.end_page,
                "page_label": p.metadata.page_label or "",
                "display_page": p.metadata.display_page or "",
                "header_path": p.metadata.header_path,
                "text": p.text,
            }
            for p in parents
        ]
        if not records:
            return

        if self._parents is None:
            self._parents = self._db.create_table(self._PARENTS_TABLE, records)
            logger.info(
                "Created LanceDB table '%s' (%d rows)",
                self._PARENTS_TABLE,
                len(records),
            )
        else:
            # Safe upsert: backup existing rows for target IDs, delete, add,
            # and restore backup on add failure.
            ids = list(dict.fromkeys(r["parent_id"] for r in records))
            existing_rows: list[dict[str, Any]] = []
            try:
                for id_batch in self._chunk(ids, self._MAX_IN_CLAUSE_VALUES):
                    existing_rows.extend(
                        self._parents
                        .search()
                        .where(self._where_in("parent_id", id_batch), prefilter=True)
                        .to_list()
                    )
                for id_batch in self._chunk(ids, self._MAX_IN_CLAUSE_VALUES):
                    self._parents.delete(self._where_in("parent_id", id_batch))
                self._parents.add(records)
            except Exception as exc:
                logger.critical(
                    "Safe upsert failed for parents table; attempting rollback. "
                    "table=%s ids_count=%d error=%s",
                    self._PARENTS_TABLE,
                    len(ids),
                    exc,
                )
                try:
                    if existing_rows:
                        self._parents.add(existing_rows)
                        logger.info(
                            "Rollback restored %d existing parent rows for table '%s'",
                            len(existing_rows),
                            self._PARENTS_TABLE,
                        )
                except Exception as rollback_exc:
                    logger.critical(
                        "Rollback failed for parents table after safe upsert error. "
                        "table=%s ids_count=%d rollback_error=%s",
                        self._PARENTS_TABLE,
                        len(ids),
                        rollback_exc,
                    )
                raise

    @staticmethod
    def _normalize_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            page = int(value)
        except (TypeError, ValueError):
            return None
        return page if page >= 1 else None

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and not (isinstance(value, str) and value == "")
        }

    @staticmethod
    def _row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
        """Extract and clean standard metadata fields from a LanceDB row."""
        meta = {
            "source_id": row.get("source_id", ""),
            "page_number": StorageEngine._normalize_positive_int(row.get("page_number")),
            "start_page": StorageEngine._normalize_positive_int(row.get("start_page")),
            "end_page": StorageEngine._normalize_positive_int(row.get("end_page")),
            "page_label": row.get("page_label"),
            "display_page": row.get("display_page"),
            "header_path": row.get("header_path", ""),
            "parent_id": row.get("parent_id", ""),
        }
        return StorageEngine._clean_metadata(meta)

    def add_children(
        self,
        children: Iterable[ChildChunk],
        *,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        child_list = list(children)
        if not child_list:
            return

        if embeddings is not None and len(embeddings) != len(child_list):
            raise ValueError("Embeddings length must match children length.")

        records: list[dict[str, Any]] = []
        for i, child in enumerate(child_list):
            record: dict[str, Any] = {
                "id": child.id,
                "text": child.text,
                "source_id": child.metadata.source_id,
                "page_number": child.metadata.page_number or 0,
                "start_page": child.metadata.start_page,
                "end_page": child.metadata.end_page,
                "page_label": child.metadata.page_label or "",
                "display_page": child.metadata.display_page or "",
                "header_path": child.metadata.header_path,
                "parent_id": child.metadata.parent_id or "",
            }
            if embeddings is not None:
                record["vector"] = embeddings[i]
            records.append(record)

        if self._table is None:
            self._table = self._db.create_table(self._table_name, records)
            self._migrate_children_schema()
            # Create FTS index on the text column for hybrid search
            self._ensure_fts_index(force_rebuild=True)
            logger.info(
                "Created LanceDB table '%s' with %d rows + FTS index",
                self._table_name, len(records),
            )
        else:
            self._table.add(records)
            self._mark_fts_dirty(len(records))
            self._maybe_rebuild_fts_after_write()
            logger.info(
                "Added %d rows to LanceDB table '%s' (FTS pending=%s)",
                len(records),
                self._table_name,
                self._fts_dirty,
            )

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        if self._parents is None:
            return None
        rows = (
            self._parents
            .search()
            .where(self._where_eq("parent_id", parent_id), prefilter=True)
            .select(["text"])
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        text = rows[0].get("text")
        return text if isinstance(text, str) else None

    def get_parent_texts(self, parent_ids: Iterable[str]) -> dict[str, str]:
        if self._parents is None:
            return {}
        unique_ids = list(dict.fromkeys(pid for pid in parent_ids if isinstance(pid, str) and pid))
        if not unique_ids:
            return {}

        result: dict[str, str] = {}
        for id_batch in self._chunk(unique_ids, self._MAX_IN_CLAUSE_VALUES):
            rows = (
                self._parents
                .search()
                .where(self._where_in("parent_id", id_batch), prefilter=True)
                .select(["parent_id", "text"])
                .to_list()
            )
            for row in rows:
                row_parent_id = row.get("parent_id")
                row_text = row.get("text")
                if isinstance(row_parent_id, str) and isinstance(row_text, str):
                    result[row_parent_id] = row_text
        return result

    def get_children_by_ids(self, ids: list[str]) -> dict[str, dict[str, object]]:
        """Fetch child chunks by their IDs from LanceDB."""
        if not ids or self._table is None:
            return {}
        _NON_VECTOR_COLS = [
            "id", "text", "source_id", "page_number",
            "start_page", "end_page", "page_label", "display_page", "header_path", "parent_id",
        ]

        unique_ids = list(dict.fromkeys(child_id for child_id in ids if child_id))
        result: dict[str, dict[str, object]] = {}
        try:
            for id_batch in self._chunk(unique_ids, self._MAX_IN_CLAUSE_VALUES):
                rows = (
                    self._table
                    .search()
                    .where(self._where_in("id", id_batch), prefilter=True)
                    .select(_NON_VECTOR_COLS)
                    .to_list()
                )
                for row in rows:
                    child_id = row.get("id")
                    if not isinstance(child_id, str):
                        continue
                    meta = self._row_to_metadata(row)
                    result[child_id] = {"text": row.get("text", ""), "metadata": meta}
        except Exception as exc:
            logger.error("Failed to fetch children by ids: %s", exc)
            return {}
        return result

    def _execute_hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str],
        bm25_weight: float,
    ) -> list[dict[str, Any]]:
        if self._table is None:
            return []

        builder = (
            self._table
            .search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
        )

        # Preserve the current default hybrid ranking behavior unless an
        # explicit non-default BM25 coefficient is requested.
        clamped_bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
        if abs(clamped_bm25_weight - 0.5) > 1e-9:
            try:
                from lancedb.rerankers import LinearCombinationReranker

                vector_weight = 1.0 - clamped_bm25_weight
                builder = builder.rerank(LinearCombinationReranker(weight=vector_weight))
            except Exception as exc:
                logger.warning(
                    "Unable to apply bm25_weight=%s; using default LanceDB hybrid fusion instead: %s",
                    clamped_bm25_weight,
                    exc,
                )

        builder = builder.limit(top_k)
        if source_id:
            builder = builder.where(self._where_eq("source_id", source_id), prefilter=True)
        return builder.to_list()

    def _run_hybrid_search_with_index_retry(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str],
        bm25_weight: float,
    ) -> list[dict[str, Any]]:
        try:
            return self._execute_hybrid_search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                source_id=source_id,
                bm25_weight=bm25_weight,
            )
        except Exception as exc:
            logger.warning(
                "Hybrid search failed once; rebuilding FTS index before retry. Error: %s",
                exc,
            )
            self._ensure_fts_index(force_rebuild=True)
            return self._execute_hybrid_search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                source_id=source_id,
                bm25_weight=bm25_weight,
            )

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str] = None,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """LanceDB native hybrid search (vector ANN + full-text BM25 with RRF fusion)."""
        if self._table is None:
            raise RuntimeError("LanceDB table is not initialized. Run ingest first.")

        self._refresh_fts_if_dirty_for_query()
        rows = self._run_hybrid_search_with_index_retry(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
            source_id=source_id,
            bm25_weight=bm25_weight,
        )

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            meta = self._row_to_metadata(row)
            results.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "metadata": meta,
                "score": float(row.get("_relevance_score", 0.0)),
                "rank": rank,
            })
        return results

    def vector_search(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Dense vector-only search used when hybrid retrieval is disabled."""
        if self._table is None:
            raise RuntimeError("LanceDB table is not initialized. Run ingest first.")

        builder = self._table.search(query_vector).limit(top_k)
        if source_id:
            builder = builder.where(self._where_eq("source_id", source_id), prefilter=True)

        rows = builder.to_list()
        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            distance_raw = row.get("_distance")
            score: float
            try:
                distance = float(distance_raw)
            except (TypeError, ValueError):
                score = float(row.get("_score", 0.0))
            else:
                score = 1.0 / (1.0 + max(0.0, distance))

            results.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "metadata": self._row_to_metadata(row),
                    "score": score,
                    "rank": rank,
                }
            )
        return results

    def list_source_ids(self) -> list[str]:
        if self._parents is None:
            return []
        rows = self._parents.to_arrow().to_pylist()
        return sorted(set(r["source_id"] for r in rows if r.get("source_id")))

    def persist_source_page_offset(
        self,
        source_id: str,
        page_offset: int = 1,
        *,
        source_path: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        citation_reference: Optional[str] = None,
    ) -> None:
        """Persist the page offset for a source, preserving existing summary/path fields.

        Creates a minimal record if no record exists yet — this handles the
        ``summarize=False`` (``--no-summarize``) case where
        ``upsert_source_summary`` is never called.
        """
        sid = source_id.strip()
        if not sid:
            raise ValueError("source_id must be non-empty.")

        existing_summary = ""
        existing_source_path = ""
        existing_snapshot_path = ""
        existing_citation_reference = ""

        if self._summaries is not None:
            rows = (
                self._summaries
                .search()
                .where(self._where_eq("source_id", sid), prefilter=True)
                .to_list()
            )
            if rows:
                r = rows[0]
                existing_summary = r.get("summary") or ""
                existing_source_path = r.get("source_path") or ""
                existing_snapshot_path = r.get("snapshot_path") or ""
                existing_citation_reference = r.get("citation_reference") or ""
                self._summaries.delete(self._where_eq("source_id", sid))

        record: dict[str, Any] = {
            "source_id": sid,
            "summary": existing_summary,
            "source_path": source_path if source_path is not None else existing_source_path,
            "snapshot_path": snapshot_path if snapshot_path is not None else existing_snapshot_path,
            "citation_reference": (
                citation_reference
                if citation_reference is not None
                else existing_citation_reference
            ),
            "page_offset": page_offset,
        }
        if self._summaries is None:
            self._summaries = self._db.create_table(self._SUMMARIES_TABLE, [record])
            logger.info(
                "Created LanceDB table '%s' via persist_source_page_offset",
                self._SUMMARIES_TABLE,
            )
        else:
            self._summaries.add([record])

    def upsert_source_summary(
        self,
        *,
        source_id: str,
        summary: str,
        source_path: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        citation_reference: Optional[str] = None,
        page_offset: int = 1,
    ) -> None:
        """Insert or update a source summary record (schema v3).

        Parameters
        ----------
        source_id : str
            Unique identifier for the source.
        summary : str
            Summary text for the source.
        source_path : str | None
            Original file path used during ingest.
        snapshot_path : str | None
            Path to cached text snapshot under data/source_cache/.
        page_offset : int
            Starting page number for the first physical PDF page (default 1).
        """
        if not source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if not summary.strip():
            raise ValueError("summary must be non-empty.")
        sid = source_id.strip()
        record: dict[str, Any] = {
            "source_id": sid,
            "summary": summary.strip(),
            "source_path": source_path or "",
            "snapshot_path": snapshot_path or "",
            "citation_reference": citation_reference or "",
            "page_offset": page_offset,
        }
        if self._summaries is None:
            self._summaries = self._db.create_table(self._SUMMARIES_TABLE, [record])
            logger.info("Created LanceDB table '%s' (schema v3)", self._SUMMARIES_TABLE)
        else:
            try:
                self._summaries.delete(self._where_eq("source_id", sid))
            except Exception as exc:
                logger.critical(
                    "Upsert delete failed for summaries table; aborting to prevent duplicates. "
                    "table=%s source_id=%s error=%s",
                    self._SUMMARIES_TABLE,
                    sid,
                    exc,
                )
                raise
            self._summaries.add([record])

    def get_source_summaries(self) -> dict[str, str]:
        if self._summaries is None:
            return {}
        rows = self._summaries.to_arrow().to_pylist()
        return {
            r["source_id"]: r["summary"]
            for r in rows
            if r.get("source_id") and r.get("summary")
        }

    def get_source_page_offsets(self) -> dict[str, int]:
        """Return page offset for each source. Absent or null values default to 1."""
        if self._summaries is None:
            return {}
        rows = self._summaries.to_arrow().to_pylist()
        return {
            r["source_id"]: int(r.get("page_offset") or 1)
            for r in rows
            if r.get("source_id")
        }

    def get_source_details(self) -> list[dict[str, Any]]:
        """Return full details for all sources (schema v3 fields).

        Returns list of dicts with keys: source_id, summary, source_path,
        snapshot_path, citation_reference, page_offset.
        """
        if self._summaries is None:
            return []
        rows = self._summaries.to_arrow().to_pylist()
        return [
            {
                "source_id": r.get("source_id", ""),
                "summary": r.get("summary", ""),
                "source_path": r.get("source_path", ""),
                "snapshot_path": r.get("snapshot_path", ""),
                "citation_reference": r.get("citation_reference", ""),
                "page_offset": int(r.get("page_offset") or 1),
            }
            for r in rows
            if r.get("source_id")
        ]

    def get_source_detail(self, source_id: str) -> Optional[dict[str, Any]]:
        """Return details for a single source, or None if not found."""
        if self._summaries is None:
            return None
        rows = (
            self._summaries
            .search()
            .where(self._where_eq("source_id", source_id), prefilter=True)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "source_id": r.get("source_id", ""),
            "summary": r.get("summary", ""),
            "source_path": r.get("source_path", ""),
            "snapshot_path": r.get("snapshot_path", ""),
            "citation_reference": r.get("citation_reference", ""),
            "page_offset": int(r.get("page_offset") or 1),
        }

    def _ensure_geo_mentions_table(self) -> Optional[lancedb.table.Table]:
        if self._geo_mentions is not None:
            self._migrate_geo_mentions_schema()
            self._ensure_geo_mentions_indexes()
            return self._geo_mentions
        try:
            self._geo_mentions = self._db.create_table(
                self._GEO_MENTIONS_TABLE,
                schema=self._GEO_MENTIONS_SCHEMA,
                exist_ok=True,
            )
            self._geo_mentions_indexes_ready = False
        except Exception as exc:
            logger.error("Failed to create/open '%s': %s", self._GEO_MENTIONS_TABLE, exc)
            return None
        self._migrate_geo_mentions_schema()
        self._ensure_geo_mentions_indexes()
        return self._geo_mentions

    def upsert_geo_mentions(self, mentions: list[dict]) -> None:
        """Batch write geo mentions. Creates table if not present."""
        if not mentions:
            return
        table = self._ensure_geo_mentions_table()
        if table is None:
            return

        records: list[dict[str, Any]] = []
        for row in mentions:
            mention_id = str(row.get("id", "")).strip()
            source_id = str(row.get("source_id", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            place_name = str(row.get("place_name", "")).strip()
            matched_input = str(row.get("matched_input", "")).strip()
            matched_on = str(row.get("matched_on", "")).strip()
            method = str(row.get("method", "")).strip()
            if not mention_id or not source_id or not chunk_id or not place_name:
                continue
            try:
                geonameid = int(row.get("geonameid"))
                lat = float(row.get("lat"))
                lon = float(row.get("lon"))
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError):
                continue

            raw_score = row.get("raw_score")
            raw_score_value: float | None
            if raw_score is None:
                raw_score_value = None
            else:
                try:
                    raw_score_value = float(raw_score)
                except (TypeError, ValueError):
                    raw_score_value = None

            margin_score = row.get("margin_score")
            margin_score_value: float | None
            if margin_score is None:
                margin_score_value = None
            else:
                try:
                    margin_score_value = float(margin_score)
                except (TypeError, ValueError):
                    margin_score_value = None

            ner_score = row.get("ner_score")
            ner_score_value: float | None
            if ner_score is None:
                ner_score_value = None
            else:
                try:
                    ner_score_value = float(ner_score)
                except (TypeError, ValueError):
                    ner_score_value = None

            geocoded_at = row.get("geocoded_at")
            geocoded_at_value: float | None
            if geocoded_at is None:
                geocoded_at_value = None
            else:
                try:
                    geocoded_at_value = float(geocoded_at)
                except (TypeError, ValueError):
                    geocoded_at_value = None

            candidate_count = row.get("candidate_count")
            try:
                candidate_count_value = max(1, int(candidate_count if candidate_count is not None else 1))
            except (TypeError, ValueError):
                candidate_count_value = 1

            entity_type = row.get("entity_type")
            entity_type_value = str(entity_type).strip() if entity_type is not None else ""
            geocoder_version = row.get("geocoder_version")
            geocoder_version_value = str(geocoder_version).strip() if geocoder_version is not None else ""

            records.append(
                {
                    "id": mention_id,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "place_name": place_name,
                    "matched_input": matched_input,
                    "matched_on": matched_on,
                    "geonameid": geonameid,
                    "lat": lat,
                    "lon": lon,
                    "confidence": confidence,
                    "method": method,
                    "raw_score": raw_score_value,
                    "is_ambiguous": bool(row.get("is_ambiguous", False)),
                    "candidate_count": candidate_count_value,
                    "margin_score": margin_score_value,
                    "entity_type": entity_type_value or None,
                    "ner_score": ner_score_value,
                    "geocoder_version": geocoder_version_value or None,
                    "geocoded_at": geocoded_at_value,
                }
            )

        if not records:
            return

        unique_ids = list(dict.fromkeys(record["id"] for record in records))
        try:
            for id_batch in self._chunk(unique_ids, self._MAX_IN_CLAUSE_VALUES):
                table.delete(self._where_in("id", id_batch))
            table.add(records)
        except Exception as exc:
            logger.error("Failed to upsert geo mentions: %s", exc)
            raise

    def get_geo_mentions(
        self,
        source_id: str | None = None,
        source_ids: Optional[list[str]] = None,
        min_confidence: float = 0.0,
        q: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """Return mentions filtered by confidence and optional source_id/source_ids union."""
        try:
            normalized_min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            raise ValueError("min_confidence must be a finite float in range [0.0, 1.0].")

        if not math.isfinite(normalized_min_confidence) or not (0.0 <= normalized_min_confidence <= 1.0):
            raise ValueError("min_confidence must be a finite float in range [0.0, 1.0].")

        if limit < 1:
            raise ValueError("limit must be >= 1.")
        if offset < 0:
            raise ValueError("offset must be >= 0.")

        normalized_single = source_id.strip() if isinstance(source_id, str) else ""
        normalized_sources: list[str] = []
        if source_ids is not None:
            for raw in source_ids:
                if raw is None:
                    continue
                sid = str(raw).strip()
                if sid and sid not in normalized_sources:
                    normalized_sources.append(sid)

        if normalized_single and normalized_single not in normalized_sources:
            normalized_sources.append(normalized_single)

        if source_ids is not None and not normalized_sources and not normalized_single:
            return []

        search_text = q.strip().lower() if isinstance(q, str) else ""

        table = self._geo_mentions
        if table is None:
            try:
                table = self._db.open_table(self._GEO_MENTIONS_TABLE)
                self._geo_mentions = table
                self._geo_mentions_indexes_ready = False
                self._migrate_geo_mentions_schema()
                self._ensure_geo_mentions_indexes()
            except ValueError:
                return []

        where_parts = [f"confidence >= {normalized_min_confidence}"]
        if normalized_sources:
            where_parts.append(self._where_in("source_id", normalized_sources))
        elif normalized_single:
            where_parts.append(self._where_eq("source_id", normalized_single))
        base_where = " AND ".join(f"({part})" for part in where_parts)
        select_columns = [
            "id",
            "source_id",
            "chunk_id",
            "place_name",
            "matched_input",
            "matched_on",
            "geonameid",
            "lat",
            "lon",
            "confidence",
            "method",
            "raw_score",
            "is_ambiguous",
            "candidate_count",
            "margin_score",
            "entity_type",
            "ner_score",
            "geocoder_version",
            "geocoded_at",
        ]

        rows: list[dict[str, Any]]
        if search_text:
            search_predicate = (
                f"({self._where_contains_ci('place_name', search_text)}) OR "
                f"({self._where_contains_ci('matched_input', search_text)})"
            )
            pushed_where = f"({base_where}) AND ({search_predicate})"
            try:
                rows = (
                    table.search()
                    .where(pushed_where, prefilter=True)
                    .select(select_columns)
                    .offset(offset)
                    .limit(limit)
                    .to_list()
                )
            except Exception as exc:
                logger.debug(
                    "Geo mention text filter pushdown unsupported; falling back to in-memory filtering: %s",
                    exc,
                )
                rows = (
                    table.search()
                    .where(base_where, prefilter=True)
                    .select(select_columns)
                    .to_list()
                )
                rows = [
                    row
                    for row in rows
                    if search_text in str(row.get("place_name", "")).lower()
                    or search_text in str(row.get("matched_input", "")).lower()
                ][offset : offset + limit]
        else:
            rows = (
                table.search()
                .where(base_where, prefilter=True)
                .select(select_columns)
                .offset(offset)
                .limit(limit)
                .to_list()
            )
        return [
            {
                "id": str(row.get("id", "")),
                "source_id": str(row.get("source_id", "")),
                "chunk_id": str(row.get("chunk_id", "")),
                "place_name": str(row.get("place_name", "")),
                "matched_input": str(row.get("matched_input", "")),
                "matched_on": str(row.get("matched_on", "")),
                "geonameid": int(row.get("geonameid")),
                "lat": float(row.get("lat")),
                "lon": float(row.get("lon")),
                "confidence": float(row.get("confidence")),
                "method": str(row.get("method", "")),
                "raw_score": float(row.get("raw_score")) if row.get("raw_score") is not None else None,
                "is_ambiguous": bool(row.get("is_ambiguous", False)),
                "candidate_count": int(row.get("candidate_count")) if row.get("candidate_count") is not None else None,
                "margin_score": float(row.get("margin_score")) if row.get("margin_score") is not None else None,
                "entity_type": str(row.get("entity_type", "")) if row.get("entity_type") is not None else None,
                "ner_score": float(row.get("ner_score")) if row.get("ner_score") is not None else None,
                "geocoder_version": str(row.get("geocoder_version", "")) if row.get("geocoder_version") is not None else None,
                "geocoded_at": float(row.get("geocoded_at")) if row.get("geocoded_at") is not None else None,
            }
            for row in rows
            if row.get("id") is not None
            and row.get("source_id") is not None
            and row.get("chunk_id") is not None
            and row.get("geonameid") is not None
            and row.get("lat") is not None
            and row.get("lon") is not None
            and row.get("confidence") is not None
        ]

    def delete_geo_mentions_by_source(self, source_id: str) -> None:
        """Called from delete_source() — cascading delete."""
        table = self._geo_mentions
        if table is None:
            try:
                table = self._db.open_table(self._GEO_MENTIONS_TABLE)
                self._geo_mentions = table
            except ValueError:
                return
        table.delete(self._where_eq("source_id", source_id))

    def delete_geo_mention(self, mention_id: str) -> None:
        """Single row delete by UUID for manual correction."""
        table = self._geo_mentions
        if table is None:
            try:
                table = self._db.open_table(self._GEO_MENTIONS_TABLE)
                self._geo_mentions = table
            except ValueError:
                return
        table.delete(self._where_eq("id", mention_id))

    def upsert_person_mentions(self, mentions: list[dict]) -> None:
        """Batch write person mentions. Creates table if not present."""
        if not mentions:
            return
        table = self._ensure_person_mentions_table()
        if table is None:
            return

        records: list[dict[str, Any]] = []
        for row in mentions:
            mention_id = str(row.get("id", "")).strip()
            source_id = str(row.get("source_id", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            raw_name = str(row.get("raw_name", "")).strip()
            canonical_name = str(row.get("canonical_name", "")).strip()
            method = str(row.get("method", "")).strip()

            if not mention_id or not source_id or not chunk_id or not raw_name or not canonical_name:
                continue
            try:
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError):
                continue

            role_hint_raw = row.get("role_hint")
            role_hint = str(role_hint_raw).strip() if role_hint_raw is not None else ""
            context_snippet_raw = row.get("context_snippet")
            context_snippet = str(context_snippet_raw).strip() if context_snippet_raw is not None else ""

            records.append(
                {
                    "id": mention_id,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "raw_name": raw_name,
                    "canonical_name": canonical_name,
                    "confidence": confidence,
                    "method": method,
                    "role_hint": role_hint or None,
                    "context_snippet": context_snippet,
                }
            )

        if not records:
            return

        unique_ids = list(dict.fromkeys(record["id"] for record in records))
        try:
            for id_batch in self._chunk(unique_ids, self._MAX_IN_CLAUSE_VALUES):
                table.delete(self._where_in("id", id_batch))
            table.add(records)
        except Exception as exc:
            logger.error("Failed to upsert person mentions: %s", exc)
            raise

    def get_person_mentions(
        self,
        source_id: str | None = None,
        source_ids: Optional[list[str]] = None,
        canonical_name: str | None = None,
        min_confidence: float = 0.0,
        q: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """Return mention-level person rows with source_id/source_ids union semantics."""
        try:
            normalized_min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            raise ValueError("min_confidence must be a finite float in range [0.0, 1.0].")

        if not math.isfinite(normalized_min_confidence) or not (0.0 <= normalized_min_confidence <= 1.0):
            raise ValueError("min_confidence must be a finite float in range [0.0, 1.0].")
        if limit < 1:
            raise ValueError("limit must be >= 1.")
        if offset < 0:
            raise ValueError("offset must be >= 0.")

        normalized_single = source_id.strip() if isinstance(source_id, str) else ""
        normalized_sources: list[str] = []
        if source_ids is not None:
            for raw in source_ids:
                if raw is None:
                    continue
                sid = str(raw).strip()
                if sid and sid not in normalized_sources:
                    normalized_sources.append(sid)

        if normalized_single and normalized_single not in normalized_sources:
            normalized_sources.append(normalized_single)

        if source_ids is not None and not normalized_sources and not normalized_single:
            return []

        normalized_canonical = canonical_name.strip() if isinstance(canonical_name, str) else ""
        if canonical_name is not None and not normalized_canonical:
            return []

        search_text = q.strip().lower() if isinstance(q, str) else ""

        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
                self._person_mentions_indexes_ready = False
                self._ensure_person_mentions_indexes()
            except ValueError:
                return []

        where_parts = [f"confidence >= {normalized_min_confidence}"]
        if normalized_sources:
            where_parts.append(self._where_in("source_id", normalized_sources))
        elif normalized_single:
            where_parts.append(self._where_eq("source_id", normalized_single))
        if normalized_canonical:
            where_parts.append(self._where_eq("canonical_name", normalized_canonical))
        base_where = " AND ".join(f"({part})" for part in where_parts)
        select_columns = [
            "id",
            "source_id",
            "chunk_id",
            "raw_name",
            "canonical_name",
            "confidence",
            "method",
            "role_hint",
            "context_snippet",
        ]

        if search_text:
            search_predicate = (
                f"({self._where_contains_ci('canonical_name', search_text)}) OR "
                f"({self._where_contains_ci('raw_name', search_text)})"
            )
            pushed_where = f"({base_where}) AND ({search_predicate})"
            try:
                rows = (
                    table.search()
                    .where(pushed_where, prefilter=True)
                    .select(select_columns)
                    .offset(offset)
                    .limit(limit)
                    .to_list()
                )
            except Exception as exc:
                logger.debug(
                    "Person mention text filter pushdown unsupported; falling back to in-memory filtering: %s",
                    exc,
                )
                # Fallback path intentionally pulls the already source/confidence-scoped
                # result set and filters in Python. This is acceptable for the expected
                # mention-table sizes in offline single-user workspaces.
                rows = (
                    table.search()
                    .where(base_where, prefilter=True)
                    .select(select_columns)
                    .to_list()
                )
                rows = [
                    row
                    for row in rows
                    if search_text in str(row.get("canonical_name", "")).lower()
                    or search_text in str(row.get("raw_name", "")).lower()
                ][offset : offset + limit]
        else:
            rows = (
                table.search()
                .where(base_where, prefilter=True)
                .select(select_columns)
                .offset(offset)
                .limit(limit)
                .to_list()
            )

        result: list[dict[str, Any]] = []
        for row in rows:
            if row.get("id") is None:
                continue
            confidence_raw = row.get("confidence")
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                continue
            result.append(
                {
                    "id": str(row.get("id", "")),
                    "source_id": str(row.get("source_id", "")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "raw_name": str(row.get("raw_name", "")),
                    "canonical_name": str(row.get("canonical_name", "")),
                    "confidence": confidence,
                    "method": str(row.get("method", "")),
                    "role_hint": str(row.get("role_hint", "")) if row.get("role_hint") is not None else None,
                    "context_snippet": str(row.get("context_snippet", "")),
                }
            )
        return result

    def get_person_mentions_by_canonical(
        self,
        canonical_name: str,
        *,
        source_id: str | None = None,
        source_ids: Optional[list[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        return self.get_person_mentions(
            source_id=source_id,
            source_ids=source_ids,
            canonical_name=canonical_name,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )

    def get_person_mention(self, mention_id: str) -> Optional[dict[str, Any]]:
        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
            except ValueError:
                return None

        rows = (
            table.search()
            .where(self._where_eq("id", mention_id), prefilter=True)
            .select([
                "id",
                "source_id",
                "chunk_id",
                "raw_name",
                "canonical_name",
                "confidence",
                "method",
                "role_hint",
                "context_snippet",
            ])
            .limit(1)
            .to_list()
        )
        if not rows:
            return None

        row = rows[0]
        confidence_raw = row.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "id": str(row.get("id", "")),
            "source_id": str(row.get("source_id", "")),
            "chunk_id": str(row.get("chunk_id", "")),
            "raw_name": str(row.get("raw_name", "")),
            "canonical_name": str(row.get("canonical_name", "")),
            "confidence": confidence,
            "method": str(row.get("method", "")),
            "role_hint": str(row.get("role_hint", "")) if row.get("role_hint") is not None else None,
            "context_snippet": str(row.get("context_snippet", "")),
        }

    def delete_person_mentions_by_source(self, source_id: str) -> None:
        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
            except ValueError:
                return
        table.delete(self._where_eq("source_id", source_id))

    def delete_person_mention(self, mention_id: str) -> None:
        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
            except ValueError:
                return
        table.delete(self._where_eq("id", mention_id))

    def merge_person_canonical_names(
        self,
        source_canonical_name: str,
        target_canonical_name: str,
    ) -> int:
        """Merge all mentions from one canonical name into another.

        Returns the number of mention rows rewritten.
        """
        source = source_canonical_name.strip()
        target = target_canonical_name.strip()
        if not source or not target:
            raise ValueError("source_canonical_name and target_canonical_name must be non-empty.")
        if source == target:
            return 0

        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
            except ValueError:
                return 0

        source_rows = (
            table.search()
            .where(self._where_eq("canonical_name", source), prefilter=True)
            .select([
                "id",
                "source_id",
                "chunk_id",
                "raw_name",
                "canonical_name",
                "confidence",
                "method",
                "role_hint",
                "context_snippet",
            ])
            .to_list()
        )
        if not source_rows:
            return 0

        rewritten: list[dict[str, Any]] = []
        mention_ids: list[str] = []
        for row in source_rows:
            mention_id = str(row.get("id", "")).strip()
            source_id = str(row.get("source_id", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            raw_name = str(row.get("raw_name", "")).strip()
            method = str(row.get("method", "")).strip()
            if not mention_id or not source_id or not chunk_id or not raw_name:
                continue

            try:
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0

            role_hint_raw = row.get("role_hint")
            role_hint = str(role_hint_raw).strip() if role_hint_raw is not None else ""
            context_snippet_raw = row.get("context_snippet")
            context_snippet = str(context_snippet_raw).strip() if context_snippet_raw is not None else ""

            mention_ids.append(mention_id)
            rewritten.append(
                {
                    "id": mention_id,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "raw_name": raw_name,
                    "canonical_name": target,
                    "confidence": confidence,
                    "method": method,
                    "role_hint": role_hint or None,
                    "context_snippet": context_snippet,
                }
            )

        if not rewritten:
            return 0

        try:
            for id_batch in self._chunk(mention_ids, self._MAX_IN_CLAUSE_VALUES):
                table.delete(self._where_in("id", id_batch))
            table.add(rewritten)
        except Exception as exc:
            logger.error(
                "Failed to merge canonical names from '%s' into '%s': %s",
                source,
                target,
                exc,
            )
            raise

        return len(rewritten)

    def list_person_mentions_for_registry(self) -> list[dict[str, Any]]:
        """Return minimal row shape used by PersonResolver warm/re-warm."""
        table = self._person_mentions
        if table is None:
            try:
                table = self._db.open_table(self._PERSON_MENTIONS_TABLE)
                self._person_mentions = table
            except ValueError:
                return []

        rows = (
            table.search()
            .select(["source_id", "raw_name", "canonical_name"])
            .to_list()
        )
        return [
            {
                "source_id": str(row.get("source_id", "")),
                "raw_name": str(row.get("raw_name", "")),
                "canonical_name": str(row.get("canonical_name", "")),
            }
            for row in rows
            if row.get("canonical_name") is not None and row.get("raw_name") is not None
        ]

    def delete_source(self, source_id: str) -> bool:
        """Delete all data for a source: children, parents, summary, geo/person mentions.

        Returns True if the source existed (had any data), False otherwise.
        Raises ``RuntimeError`` if any individual delete step fails so
        callers know the operation was only partially completed.
        """
        sid = source_id.strip()
        if not sid:
            raise ValueError("source_id must be non-empty.")

        deleted_any = False
        errors: list[str] = []

        # Delete child chunks
        if self._table is not None:
            try:
                self._table.delete(self._where_eq("source_id", sid))
                self._mark_fts_dirty(0)
                deleted_any = True
                logger.info("Deleted children for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete children for source '%s': %s", sid, exc)
                errors.append(f"children: {exc}")

        # Delete parent chunks
        if self._parents is not None:
            try:
                self._parents.delete(self._where_eq("source_id", sid))
                deleted_any = True
                logger.info("Deleted parents for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete parents for source '%s': %s", sid, exc)
                errors.append(f"parents: {exc}")

        # Delete summary
        if self._summaries is not None:
            try:
                self._summaries.delete(self._where_eq("source_id", sid))
                deleted_any = True
                logger.info("Deleted summary for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete summary for source '%s': %s", sid, exc)
                errors.append(f"summary: {exc}")

        # Entity mention tables are not DB-enforced foreign-key cascades, so
        # source deletes must explicitly clear both geo/person side tables.
        # Keep these calls paired with parent/child/summary deletion.
        # Delete geo mentions
        try:
            self.delete_geo_mentions_by_source(sid)
            logger.info("Deleted geo mentions for source '%s'", sid)
        except Exception as exc:
            logger.error("Failed to delete geo mentions for source '%s': %s", sid, exc)
            errors.append(f"geo_mentions: {exc}")

        # Delete person mentions
        try:
            self.delete_person_mentions_by_source(sid)
            logger.info("Deleted person mentions for source '%s'", sid)
        except Exception as exc:
            logger.error("Failed to delete person mentions for source '%s': %s", sid, exc)
            errors.append(f"person_mentions: {exc}")

        if errors:
            raise RuntimeError(
                f"Partial delete failure for source '{sid}': {'; '.join(errors)}"
            )

        return deleted_any

    def get_parent_texts_by_source(self, *, source_id: str) -> list[str]:
        if self._parents is None:
            return []
        rows = (
            self._parents
            .search()
            .where(self._where_eq("source_id", source_id), prefilter=True)
            .select(["text", "page_number"])
            .to_list()
        )
        rows.sort(key=lambda r: r.get("page_number", 0))
        return [r["text"] for r in rows if r.get("text")]
