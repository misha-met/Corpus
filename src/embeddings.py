"""MLX-native embedding model wrapper for Qwen3-Embedding.

Architecture
~~~~~~~~~~~~
- ``MlxEmbeddingModel`` lazy-loads the model on first ``encode()`` call and
  keeps it resident for the lifetime of the engine.

- **Pooling**: Qwen3-Embedding is a *decoder* (causal) model that requires
  **last-token pooling** — the embedding is taken from the final non-padding
  token, not a mean over all tokens.  Mean-pooling produces incorrect vectors
  for this architecture.

- **Padding side**: Left-padding is required so that the last column of every
  sequence always corresponds to a real token (not a pad token).  The
  tokenizer's ``padding_side`` is set to ``"left"`` immediately after load.

- **Instruction prefixes**: Qwen3-Embedding is instruction-aware.  Queries
  should be wrapped as ``"Instruct: {task}\nQuery:{query}"``.  Documents are
  encoded without any prefix.  Per-intent task strings are provided for all
  twelve Corpus research intents, giving a 1–5 % retrieval quality uplift
  compared to bare queries.

- Thread-safe lazy initialisation via double-checked locking so the model can
  be used safely from a FastAPI async/thread-pool context.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-intent task instructions
# ---------------------------------------------------------------------------
# Qwen3-Embedding is instruction-aware: wrapping queries with a task string
# consistently improves retrieval quality by 1–5 % on MTEB benchmarks.
# Documents are *never* wrapped — asymmetric encoding is intentional.
# Instructions should be written in English even for multilingual corpora.

_INTENT_TASKS: dict[str, str] = {
    # Generic fallback used when prompt_name="query" with no specific intent
    "query": (
        "Given a humanities research query, retrieve relevant document "
        "passages that answer the query"
    ),
    "overview": (
        "Given a research document query, retrieve passages that provide "
        "a broad overview of the document's content and central themes"
    ),
    "summarize": (
        "Given a summarisation request, retrieve passages that collectively "
        "cover the key points, arguments, and conclusions of the source material"
    ),
    "explain": (
        "Given a concept or term to explain, retrieve passages that define, "
        "elaborate on, and provide contextual examples of the concept"
    ),
    "analyze": (
        "Given an analytical research query, retrieve passages that provide "
        "evidence, argument, and interpretive material for in-depth analysis"
    ),
    "compare": (
        "Given a comparative research query, retrieve passages that discuss "
        "both subjects being compared and highlight similarities or differences"
    ),
    "critique": (
        "Given a critical research query, retrieve passages relevant to "
        "evaluating the strengths, weaknesses, and limitations of the subject"
    ),
    "factual": (
        "Given a specific factual question, retrieve passages that contain "
        "the direct answer along with supporting context"
    ),
    "collection": (
        "Given a query about which sources cover a topic, retrieve passages "
        "from across documents that mention or substantively discuss the topic"
    ),
    "extract": (
        "Given an extraction query, retrieve passages that contain the "
        "specific entities, dates, names, or information items to be extracted"
    ),
    "timeline": (
        "Given a temporal or historical research query, retrieve passages "
        "containing dates, events, and chronological information"
    ),
    "how_to": (
        "Given a procedural question, retrieve passages that explain the "
        "steps, methods, or processes for performing the task"
    ),
    "quote_evidence": (
        "Given a query seeking quotations or textual evidence, retrieve "
        "passages containing direct quotes or cited statements on the topic"
    ),
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MlxEmbeddingModel:
    """MLX-native embedding wrapper for Qwen3-Embedding-0.6B (and variants).

    Parameters
    ----------
    model_id:
        HuggingFace / local model path passed to ``mlx_lm.load()``.
    batch_size:
        Default number of texts per forward pass.
    max_length:
        Tokenizer truncation length in tokens.
    """

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
        self._mx: Any = None                    # cached mlx.core reference
        self._embedding_dim: Optional[int] = None
        self._resolved_backbone: Any = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> Optional[int]:
        """Embedding dimensionality; ``None`` until the model is loaded."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the model with double-checked thread-safe locking."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:          # re-check after acquiring lock
                return

            import mlx.core as mx
            import mlx_lm

            self._mx = mx
            logger.info("Loading MLX embedding model: %s", self._model_id)
            self._model, self._tokenizer = mlx_lm.load(self._model_id)
            self._resolved_backbone = self._resolve_backbone()

            # Qwen3-Embedding is a decoder model.  Left-padding is required so
            # the last column always holds a real (non-pad) token.
            hf_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
            for tok in (self._tokenizer, hf_tok):
                if hasattr(tok, "padding_side"):
                    tok.padding_side = "left"
            logger.debug("padding_side='left' set on tokenizer.")

            # Probe embedding dim at load time so callers (e.g. LanceDB index
            # creation) can read it before the first real encode() call.
            try:
                dummy_ids = self._tokenizer.encode("hello", add_special_tokens=True)
                probe = mx.array([dummy_ids[: min(8, len(dummy_ids))]])
                hidden = self._run_backbone(probe)
                mx.eval(hidden)
                self._embedding_dim = int(hidden.shape[-1])
                logger.debug("Embedding dim probed at load: %d", self._embedding_dim)
            except Exception as exc:             # noqa: BLE001
                raise RuntimeError(
                    f"Embedding backbone probe failed for model '{self._model_id}': {exc}"
                ) from exc

    def unload(self) -> None:
        """Release the model, tokenizer, and MLX reference to free memory."""
        self._model = None
        self._tokenizer = None
        self._mx = None
        self._embedding_dim = None
        self._resolved_backbone = None
        logger.info("Unloaded MLX embedding model: %s", self._model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backbone(self) -> Any:
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded.")
        for attr in ("model", "transformer", "encoder"):
            if hasattr(self._model, attr):
                backbone = getattr(self._model, attr)
                if callable(backbone):
                    return backbone
                raise RuntimeError(
                    f"Resolved embedding backbone '{attr}' is not callable on {type(self._model).__name__}."
                )
        raise RuntimeError(
            f"Could not resolve a trusted embedding backbone on {type(self._model).__name__}. "
            "Expected one of: model, transformer, encoder."
        )

    def _get_backbone(self) -> Any:
        if self._resolved_backbone is None:
            self._resolved_backbone = self._resolve_backbone()
        return self._resolved_backbone

    def _coerce_hidden(self, output: Any) -> Any:
        hidden = output
        if isinstance(hidden, (list, tuple)):
            if not hidden:
                raise RuntimeError("Embedding backbone returned an empty sequence.")
            hidden = hidden[0]
        shape = getattr(hidden, "shape", None)
        if shape is None or len(shape) != 3:
            raise RuntimeError(
                f"Embedding backbone returned invalid shape: {shape}. Expected (batch, seq, dim)."
            )
        return hidden

    def _run_backbone(self, input_ids: Any) -> Any:
        backbone = self._get_backbone()
        output = backbone(input_ids)
        return self._coerce_hidden(output)

    def _apply_instruction(self, text: str, task: str) -> str:
        """Wrap a query with a Qwen3-Embedding instruction prefix.

        Official format: ``"Instruct: {task}\nQuery:{query}"``
        Note the absence of a space after ``Query:`` — this matches the
        training data format used by Qwen3-Embedding exactly.
        """
        return f"Instruct: {task}\nQuery:{text}"

    def _tokenize_batch(
        self, texts: list[str], pad_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # TokenizerWrapper does not implement __call__ — reach past it to the
        # raw HuggingFace tokenizer at ._tokenizer, which is callable.
        hf_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)

        try:
            tokenized = hf_tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors=None,
            )
            token_rows = tokenized["input_ids"]
            mask_rows = tokenized.get("attention_mask", None)
        except (TypeError, AttributeError, ValueError) as exc:
            logger.warning(
                "Batch tokenization failed (%s); falling back to per-text encoding.",
                exc,
            )
            token_rows = [
                self._tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self._max_length,
                )
                for text in texts
            ]
            mask_rows = None

        if not token_rows:
            return (
                np.zeros((0, 0), dtype=np.int32),
                np.zeros((0, 0), dtype=np.float32),
            )

        if mask_rows is not None:
            try:
                return (
                    np.array(token_rows, dtype=np.int32),
                    np.array(mask_rows, dtype=np.float32),
                )
            except ValueError:
                pass  # ragged rows, fall through to manual padding

        # Manual left-padding (fallback path only)
        max_len = max((len(row) for row in token_rows), default=1)
        max_len = max(max_len, 1)
        input_ids = np.full((len(token_rows), max_len), pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(token_rows), max_len), dtype=np.float32)
        for idx, row in enumerate(token_rows):
            if not row:
                continue
            seq = list(row[:max_len])
            seq_len = len(seq)
            input_ids[idx, max_len - seq_len :] = np.asarray(seq, dtype=np.int32)
            attention_mask[idx, max_len - seq_len :] = 1.0
        return input_ids, attention_mask

    def _last_token_pool(
        self, hidden: Any, attention_mask: np.ndarray
    ) -> Any:
        """Extract the embedding from the last non-padding token.

        Qwen3-Embedding is a decoder model trained with last-token pooling.
        With left-padding the final column is always a real token, so we take
        ``hidden[:, -1, :]`` directly.  A right-padding fallback is retained
        in case the tokenizer padding_side could not be set.
        """
        mx = self._mx
        mask_mx = mx.array(attention_mask)

        left_padding = int(mask_mx[:, -1].sum()) == attention_mask.shape[0]
        if left_padding:
            return hidden[:, -1, :]

        # Right-padding fallback — indicates a configuration problem.
        logger.warning(
            "Right-padding detected in _last_token_pool.  "
            "Ensure tokenizer.padding_side is set to 'left' for correct "
            "Qwen3-Embedding behaviour."
        )
        seq_lengths = mask_mx.sum(axis=1).astype(mx.int32) - 1
        return mx.stack(
            [hidden[i, int(seq_lengths[i])] for i in range(hidden.shape[0])]
        )

    def _embed_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        self._ensure_loaded()
        mx = self._mx

        pad_id: int = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", 0)
        if pad_id is None:
            pad_id = 0

        input_ids_np, attention_mask = self._tokenize_batch(texts, int(pad_id))
        if input_ids_np.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        input_ids = mx.array(input_ids_np)
        hidden = self._run_backbone(input_ids)

        # Last-token pooling — required for Qwen3-Embedding (decoder architecture).
        pooled = self._last_token_pool(hidden, attention_mask)
        pooled = pooled.astype(mx.float32)

        if normalize:
            norms = mx.sqrt(mx.sum(pooled * pooled, axis=1, keepdims=True))
            norms = mx.maximum(norms, mx.array(1e-12))
            pooled = pooled / norms

        # Single fused mx.eval(): MLX builds a lazy compute graph, so deferring
        # evaluation until here allows it to optimise the full pooling +
        # normalisation pass as one operation.
        mx.eval(pooled)

        try:
            pooled_np = np.array(pooled, copy=False)
        except Exception:                        # noqa: BLE001
            pooled_np = np.asarray(pooled.tolist(), dtype=np.float32)

        if pooled_np.ndim != 2:
            raise RuntimeError(
                f"Embedding pooling produced unexpected shape {pooled_np.shape}. "
                "Expected (batch_size, embedding_dim)."
            )

        self._embedding_dim = int(pooled_np.shape[1])
        return pooled_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[list[str], str],
        normalize_embeddings: bool = True,
        batch_size: Optional[int] = None,
        prompt_name: Optional[str] = None,
        task_description: Optional[str] = None,
        return_numpy: bool = False,
        **_: Any,
    ) -> Union[list[list[float]], np.ndarray]:
        """Encode texts into embedding vectors.

        Parameters
        ----------
        texts:
            A single string or list of strings to embed.
        normalize_embeddings:
            L2-normalise output vectors.  Recommended for cosine similarity
            and dot-product ANN search (LanceDB default).
        batch_size:
            Override the instance batch size for this call only.
        prompt_name:
            Apply a task instruction prefix to each text (queries only —
            documents should be encoded without a prefix).  Pass ``"query"``
            for the generic retrieval instruction, or any of the twelve Corpus
            intent keys (e.g. ``"factual"``, ``"analyze"``, ``"timeline"``)
            for a specialised instruction that better matches the query intent.
            Ignored for document ingestion.
        task_description:
            Provide a fully custom instruction string.  Takes precedence over
            ``prompt_name`` when both are supplied.
        return_numpy:
            Return an ``np.ndarray`` of shape ``(N, D)`` instead of
            ``list[list[float]]``.  Strongly preferred for FAISS, LanceDB, and
            NumPy callers — it avoids a redundant Python object expansion that
            those libraries immediately reverse.

        Returns
        -------
        list[list[float]] or np.ndarray
            Shape ``(N, D)`` where N = len(texts) and D = embedding_dim.
        """
        if isinstance(texts, str):
            batch_texts = [texts]
        else:
            batch_texts = [str(t) for t in texts]

        if not batch_texts:
            empty = np.zeros((0, 0), dtype=np.float32)
            return empty if return_numpy else []

        # ------------------------------------------------------------------
        # Input sanitisation
        # Empty or whitespace-only strings tokenise to near-empty sequences
        # dominated by special/padding tokens and produce degenerate embeddings
        # that corrupt nearest-neighbour search.  This is almost always a bug
        # in the upstream chunking pipeline.
        # ------------------------------------------------------------------
        cleaned: list[str] = []
        for t in batch_texts:
            stripped = t.strip()
            if not stripped:
                logger.warning(
                    "Empty or whitespace-only input detected during encode(). "
                    "Replacing with '[empty]'. Check the chunking pipeline for "
                    "zero-length or whitespace-only chunks."
                )
                cleaned.append("[empty]")
            else:
                cleaned.append(stripped)

        # ------------------------------------------------------------------
        # Apply query instruction prefix (queries only; documents get none)
        # ------------------------------------------------------------------
        if task_description:
            cleaned = [self._apply_instruction(t, task_description) for t in cleaned]
        elif prompt_name:
            task = _INTENT_TASKS.get(prompt_name.lower(), _INTENT_TASKS["query"])
            cleaned = [self._apply_instruction(t, task) for t in cleaned]

        # ------------------------------------------------------------------
        # Batched inference
        # ------------------------------------------------------------------
        effective_batch_size = max(1, int(batch_size or self._batch_size))
        total = len(cleaned)
        num_batches = -(-total // effective_batch_size)   # ceiling division
        batch_arrays: list[np.ndarray] = []

        for i, start in enumerate(range(0, total, effective_batch_size)):
            end = start + effective_batch_size
            if total > effective_batch_size:
                logger.debug("Encoding batch %d/%d", i + 1, num_batches)
            batch_arrays.append(
                self._embed_batch(cleaned[start:end], normalize=normalize_embeddings)
            )

        result = (
            np.vstack(batch_arrays)
            if batch_arrays
            else np.zeros((0, 0), dtype=np.float32)
        )
        return result if return_numpy else result.tolist()
