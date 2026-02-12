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


class StorageEngine:
    """Unified storage engine backed entirely by LanceDB."""

    _PARENTS_TABLE = "parent_chunks"
    _SUMMARIES_TABLE = "source_summaries"

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        config.lance_dir.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(config.lance_dir))
        self._table_name = config.lance_table

        # --- child chunks (vectors + FTS) ---
        self._table: Optional[lancedb.table.Table] = None
        try:
            self._table = self._db.open_table(self._table_name)
            logger.info("Opened LanceDB table '%s'", self._table_name)
        except Exception:
            logger.info(
                "LanceDB table '%s' not found; will create on first ingest",
                self._table_name,
            )

        # --- parent chunks (no vectors) ---
        self._parents: Optional[lancedb.table.Table] = None
        try:
            self._parents = self._db.open_table(self._PARENTS_TABLE)
        except Exception:
            pass

        # --- source summaries (no vectors) ---
        self._summaries: Optional[lancedb.table.Table] = None
        try:
            self._summaries = self._db.open_table(self._SUMMARIES_TABLE)
        except Exception:
            pass

    def close(self) -> None:
        pass  # LanceDB connections do not require explicit close

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
            # Upsert: remove existing parent_ids then re-add
            ids = [r["parent_id"] for r in records]
            id_list = ", ".join(f"'{pid}'" for pid in ids)
            try:
                self._parents.delete(f"parent_id IN ({id_list})")
            except Exception:
                pass
            self._parents.add(records)

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
            self._table.create_fts_index("text", replace=True)
            logger.info(
                "Created LanceDB table '%s' with %d rows + FTS index",
                self._table_name, len(records),
            )
        else:
            self._table.add(records)
            # Rebuild FTS index to include new data
            self._table.create_fts_index("text", replace=True)
            logger.info("Added %d rows to LanceDB table '%s' + rebuilt FTS index", len(records), self._table_name)

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        if self._parents is None:
            return None
        for row in self._parents.to_arrow().to_pylist():
            if row["parent_id"] == parent_id:
                return row["text"]
        return None

    def get_children_by_ids(self, ids: list[str]) -> dict[str, dict[str, object]]:
        """Fetch child chunks by their IDs from LanceDB."""
        if not ids or self._table is None:
            return {}
        id_set = set(ids)
        _NON_VECTOR_COLS = [
            "id", "text", "source_id", "page_number",
            "page_label", "display_page", "header_path", "parent_id",
        ]
        try:
            rows = self._table.to_arrow().select(_NON_VECTOR_COLS).to_pylist()
        except Exception:
            return {}
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            child_id = row.get("id")
            if child_id in id_set:
                meta = {
                    "source_id": row.get("source_id", ""),
                    "page_number": row.get("page_number"),
                    "page_label": row.get("page_label"),
                    "display_page": row.get("display_page"),
                    "header_path": row.get("header_path", ""),
                    "parent_id": row.get("parent_id", ""),
                }
                meta = {k: v for k, v in meta.items() if v is not None and v != "" and v != 0}
                result[child_id] = {"text": row.get("text", ""), "metadata": meta}
        return result

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

        builder = (
            self._table
            .search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
            .limit(top_k)
        )
        if source_id:
            builder = builder.where(f"source_id = '{source_id}'", prefilter=True)

        rows = builder.to_list()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            meta = {
                "source_id": row.get("source_id", ""),
                "page_number": row.get("page_number"),
                "page_label": row.get("page_label"),
                "display_page": row.get("display_page"),
                "header_path": row.get("header_path", ""),
                "parent_id": row.get("parent_id", ""),
            }
            meta = {k: v for k, v in meta.items() if v is not None and v != "" and v != 0}
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

    def upsert_source_summary(self, *, source_id: str, summary: str) -> None:
        if not source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if not summary.strip():
            raise ValueError("summary must be non-empty.")
        sid = source_id.strip()
        record = {"source_id": sid, "summary": summary.strip()}
        if self._summaries is None:
            self._summaries = self._db.create_table(self._SUMMARIES_TABLE, [record])
            logger.info("Created LanceDB table '%s'", self._SUMMARIES_TABLE)
        else:
            try:
                self._summaries.delete(f"source_id = '{sid}'")
            except Exception:
                pass
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

    def get_parent_texts_by_source(self, *, source_id: str) -> list[str]:
        if self._parents is None:
            return []
        rows = [
            r for r in self._parents.to_arrow().to_pylist()
            if r.get("source_id") == source_id
        ]
        rows.sort(key=lambda r: r.get("page_number", 0))
        return [r["text"] for r in rows if r.get("text")]
