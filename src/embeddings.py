from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MlxEmbeddingModel:
    """MLX-native embedding wrapper with SentenceTransformer-like encode API."""

    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._batch_size = max(1, int(batch_size))
        self._max_length = max(32, int(max_length))
        self._model: Any = None
        self._tokenizer: Any = None
        self._embedding_dim: Optional[int] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import mlx_lm

        logger.info("Loading MLX embedding model: %s", self._model_id)
        self._model, self._tokenizer = mlx_lm.load(self._model_id)

    def _tokenize_batch(self, texts: list[str], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            tokenized = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            token_rows = tokenized["input_ids"]
        except Exception:
            token_rows = [
                self._tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self._max_length,
                )
                for text in texts
            ]

        if not token_rows:
            return (
                np.zeros((0, 0), dtype=np.int32),
                np.zeros((0, 0), dtype=np.float32),
            )

        max_len = max(len(row) for row in token_rows)
        if max_len <= 0:
            max_len = 1

        input_ids = np.full((len(token_rows), max_len), pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(token_rows), max_len), dtype=np.float32)

        for idx, row in enumerate(token_rows):
            if not row:
                continue
            seq = row[:max_len]
            seq_len = len(seq)
            input_ids[idx, :seq_len] = np.asarray(seq, dtype=np.int32)
            attention_mask[idx, :seq_len] = 1.0

        return input_ids, attention_mask

    def _embed_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        self._ensure_loaded()

        import mlx.core as mx

        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", 0)
        if pad_id is None:
            pad_id = 0

        input_ids_np, attention_mask = self._tokenize_batch(texts, int(pad_id))
        if input_ids_np.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        input_ids = mx.array(input_ids_np)

        backbone = getattr(self._model, "model", self._model)
        hidden = backbone(input_ids)
        mask_mx = mx.array(attention_mask)
        denom = mx.maximum(mask_mx.sum(axis=1, keepdims=True), mx.array(1e-12))
        pooled = (hidden * mask_mx[..., None]).sum(axis=1) / denom
        pooled = pooled.astype(mx.float32)
        mx.eval(pooled)

        if normalize:
            norms = mx.sqrt(mx.sum(pooled * pooled, axis=1, keepdims=True))
            norms = mx.maximum(norms, mx.array(1e-12))
            pooled = pooled / norms
            mx.eval(pooled)

        try:
            pooled_np = np.array(pooled, copy=False)
        except Exception:
            pooled_np = np.asarray(pooled.tolist(), dtype=np.float32)

        if pooled_np.ndim != 2:
            raise RuntimeError("MLX embedding pooling produced invalid shape.")

        self._embedding_dim = int(pooled_np.shape[1])
        return pooled_np

    def encode(
        self,
        texts: list[str] | str,
        normalize_embeddings: bool = True,
        batch_size: Optional[int] = None,
        **_: Any,
    ) -> list[list[float]]:
        if isinstance(texts, str):
            batch_texts = [texts]
        else:
            batch_texts = [str(text) for text in texts]

        if not batch_texts:
            return []

        effective_batch_size = max(1, int(batch_size or self._batch_size))
        vectors: list[list[float]] = []

        for start in range(0, len(batch_texts), effective_batch_size):
            end = start + effective_batch_size
            pooled = self._embed_batch(batch_texts[start:end], normalize=normalize_embeddings)
            vectors.extend(pooled.astype(np.float32).tolist())

        return vectors
