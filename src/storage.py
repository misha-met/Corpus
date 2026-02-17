from __future__ import annotations

import logging
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
    fts_rebuild_policy: str = "deferred"
    fts_rebuild_batch_size: int = 0


class StorageEngine:
    """Unified storage engine backed entirely by LanceDB."""

    _PARENTS_TABLE = "parent_chunks"
    _SUMMARIES_TABLE = "source_summaries"
    _MAX_IN_CLAUSE_VALUES = 256

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
        except ValueError:
            pass  # Table does not exist yet

        # --- source summaries (no vectors) ---
        self._summaries: Optional[lancedb.table.Table] = None
        try:
            self._summaries = self._db.open_table(self._SUMMARIES_TABLE)
            self._migrate_summaries_schema()
        except ValueError:
            pass  # Table does not exist yet

    def close(self) -> None:
        pass  # LanceDB connections do not require explicit close

    # ------------------------------------------------------------------
    # Schema migrations
    # ------------------------------------------------------------------

    _SUMMARIES_V2_COLUMNS = ("source_path", "snapshot_path")

    def _migrate_summaries_schema(self) -> None:
        """Add v2 columns to the source_summaries table if missing."""
        if self._summaries is None:
            return
        existing = set(self._summaries.schema.names)
        missing = [
            pa.field(col, pa.utf8())
            for col in self._SUMMARIES_V2_COLUMNS
            if col not in existing
        ]
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
        for table_name in (self._table_name, self._PARENTS_TABLE, self._SUMMARIES_TABLE):
            try:
                self._db.drop_table(table_name)
                logger.warning("Dropped LanceDB table '%s'", table_name)
            except Exception as exc:
                logger.debug("Could not drop table '%s': %s", table_name, exc)

        self._table = None
        self._parents = None
        self._summaries = None
        self._fts_dirty = False
        self._pending_fts_rows = 0

    def add_parents(self, parents: Iterable[ParentChunk]) -> None:
        records = [
            {
                "parent_id": p.id,
                "source_id": p.metadata.source_id,
                "page_number": p.metadata.page_number or 0,
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
            "page_number": row.get("page_number"),
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
            "page_label", "display_page", "header_path", "parent_id",
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
    ) -> list[dict[str, Any]]:
        if self._table is None:
            return []

        builder = (
            self._table
            .search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
            .limit(top_k)
        )
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
    ) -> list[dict[str, Any]]:
        try:
            return self._execute_hybrid_search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                source_id=source_id,
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
            )

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_id: Optional[str] = None,
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

    def list_source_ids(self) -> list[str]:
        if self._parents is None:
            return []
        rows = self._parents.to_arrow().to_pylist()
        return sorted(set(r["source_id"] for r in rows if r.get("source_id")))

    def upsert_source_summary(
        self,
        *,
        source_id: str,
        summary: str,
        source_path: Optional[str] = None,
        snapshot_path: Optional[str] = None,
    ) -> None:
        """Insert or update a source summary record (schema v2).

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
        }
        if self._summaries is None:
            self._summaries = self._db.create_table(self._SUMMARIES_TABLE, [record])
            logger.info("Created LanceDB table '%s' (schema v2)", self._SUMMARIES_TABLE)
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

    def get_source_details(self) -> list[dict[str, Any]]:
        """Return full details for all sources (schema v2 fields).

        Returns list of dicts with keys: source_id, summary, source_path, snapshot_path.
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
        }

    def delete_source(self, source_id: str) -> bool:
        """Delete all data for a source: children, parents, and summary.

        Returns True if the source existed (had any data), False otherwise.
        """
        sid = source_id.strip()
        if not sid:
            raise ValueError("source_id must be non-empty.")

        deleted_any = False

        # Delete child chunks
        if self._table is not None:
            try:
                self._table.delete(self._where_eq("source_id", sid))
                self._mark_fts_dirty(0)
                deleted_any = True
                logger.info("Deleted children for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete children for source '%s': %s", sid, exc)

        # Delete parent chunks
        if self._parents is not None:
            try:
                self._parents.delete(self._where_eq("source_id", sid))
                deleted_any = True
                logger.info("Deleted parents for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete parents for source '%s': %s", sid, exc)

        # Delete summary
        if self._summaries is not None:
            try:
                self._summaries.delete(self._where_eq("source_id", sid))
                deleted_any = True
                logger.info("Deleted summary for source '%s'", sid)
            except Exception as exc:
                logger.error("Failed to delete summary for source '%s': %s", sid, exc)

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
