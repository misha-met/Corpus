from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


class MlxGenerator:
    def __init__(self, model_path: str) -> None:
        self._model_id = model_path
        try:
            from mlx_lm import load
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm is not available. Install mlx-lm to continue.") from exc

        try:
            self._model, self._tokenizer = load(model_path)
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}.") from exc

    @staticmethod
    def _infer_model_size_b(model_id: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _build_repetition_penalty_processor(penalty: float):
        if penalty <= 1.0:
            return None

        def _processor(tokens, logits):
            try:
                import mlx.core as mx
            except Exception:
                return logits

            token_ids = tokens.tolist()
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            if not token_ids:
                return logits

            logits_np = mx.array(logits)
            token_ids = sorted(set(int(t) for t in token_ids))
            cols = mx.take(logits_np, mx.array(token_ids), axis=-1)
            adjusted = mx.where(
                cols > 0,
                cols / penalty,
                cols * penalty,
            )
            try:
                import numpy as np

                logits_np_np = logits_np.astype("float32").tolist()
                adjusted_np = adjusted.astype("float32").tolist()
                for col_idx, token_id in enumerate(token_ids):
                    logits_np_np[0][token_id] = adjusted_np[0][col_idx]
                return mx.array(logits_np_np)
            except Exception:
                return logits_np

        return _processor

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        cfg = config or GenerationConfig()
        model_size = self._infer_model_size_b(self._model_id)
        max_tokens = cfg.max_tokens
        repetition_penalty = cfg.repetition_penalty
        temperature = cfg.temperature
        top_p = cfg.top_p

        if model_size is not None:
            if model_size < 30:
                max_tokens = max_tokens or 140
                repetition_penalty = repetition_penalty or 1.25
                temperature = temperature or 0.05
                top_p = top_p or 0.7
            elif model_size >= 70:
                repetition_penalty = repetition_penalty or 1.05
                temperature = temperature or 0.2
                top_p = top_p or 0.9
            else:
                max_tokens = max_tokens or 400
                repetition_penalty = repetition_penalty or 1.1
                temperature = temperature or 0.15
                top_p = top_p or 0.9
        else:
            temperature = temperature or 0.2
            top_p = top_p or 0.9
        try:
            from mlx_lm import generate
            from mlx_lm.generate import make_sampler
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generate is not available.") from exc

        try:
            sampler = make_sampler(temp=temperature, top_p=top_p)
            logits_processors = []
            if repetition_penalty is not None:
                processor = self._build_repetition_penalty_processor(repetition_penalty)
                if processor is not None:
                    logits_processors.append(processor)

            return generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens or 512,
                sampler=sampler,
                logits_processors=logits_processors or None,
            )
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generation failed.") from exc
