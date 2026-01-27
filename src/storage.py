from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from .models import ChildChunk, ParentChunk


@dataclass
class StorageConfig:
    sqlite_path: Path
    chroma_dir: Path
    chroma_collection: str = "child_chunks"


class StorageEngine:
    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        config.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(config.sqlite_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                page_number INTEGER,
                header_path TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

        settings = Settings(anonymized_telemetry=False, is_persistent=True)
        self._chroma = chromadb.PersistentClient(path=str(config.chroma_dir), settings=settings)
        self._collection = self._chroma.get_or_create_collection(config.chroma_collection)

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_ids: list[str] = []

    def close(self) -> None:
        self._conn.close()

    def add_parents(self, parents: Iterable[ParentChunk]) -> None:
        rows = [
            (
                parent.id,
                parent.metadata.source_id,
                parent.metadata.page_number,
                parent.metadata.header_path,
                parent.text,
            )
            for parent in parents
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks
                    (parent_id, source_id, page_number, header_path, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def add_children(
        self,
        children: Iterable[ChildChunk],
        *,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        child_list = list(children)
        if not child_list:
            return

        ids = [child.id for child in child_list]
        documents = [child.text for child in child_list]
        metadatas = [
            {
                "source_id": child.metadata.source_id,
                "page_number": child.metadata.page_number,
                "header_path": child.metadata.header_path,
                "parent_id": child.metadata.parent_id,
            }
            for child in child_list
        ]

        if embeddings is not None and len(embeddings) != len(child_list):
            raise ValueError("Embeddings length must match children length.")

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        tokenized = [doc.split() for doc in documents]
        self._bm25_corpus.extend(tokenized)
        self._bm25_ids.extend(ids)
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        cursor = self._conn.execute(
            "SELECT text FROM parent_chunks WHERE parent_id = ?",
            (parent_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_children_by_ids(self, ids: list[str]) -> dict[str, dict[str, object]]:
        if not ids:
            return {}
        response = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        response_ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        return {
            child_id: {
                "text": doc,
                "metadata": meta or {},
            }
            for child_id, doc, meta in zip(response_ids, documents, metadatas)
        }

    def query_children(self, *, embeddings: list[list[float]], top_k: int) -> dict[str, list]:
        return self._collection.query(
            query_embeddings=embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def persist_bm25(self, path: Path) -> None:
        payload = {
            "ids": self._bm25_ids,
            "corpus": self._bm25_corpus,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load_bm25(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._bm25_ids = list(payload.get("ids", []))
        self._bm25_corpus = list(payload.get("corpus", []))
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    @property
    def bm25(self) -> Optional[BM25Okapi]:
        return self._bm25

    @property
    def bm25_ids(self) -> list[str]:
        return list(self._bm25_ids)
