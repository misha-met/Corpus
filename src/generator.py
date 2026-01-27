from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9


class MlxGenerator:
    def __init__(self, model_path: str) -> None:
        try:
            from mlx_lm import load
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm is not available. Install mlx-lm to continue.") from exc

        try:
            self._model, self._tokenizer = load(model_path)
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}.") from exc

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        cfg = config or GenerationConfig()
        try:
            from mlx_lm import generate
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generate is not available.") from exc

        try:
            return generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=cfg.max_tokens,
                temp=cfg.temperature,
                top_p=cfg.top_p,
            )
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generation failed.") from exc
