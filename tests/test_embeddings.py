from __future__ import annotations

import pytest

from src.embeddings import MlxEmbeddingModel


class _GoodBackbone:
    def __call__(self, _input_ids):
        class _Hidden:
            shape = (1, 4, 8)

        return _Hidden()


class _BadBackbone:
    def __call__(self, _input_ids):
        class _Hidden:
            shape = (1, 8)

        return _Hidden()


class _ModelWithTrustedAttr:
    def __init__(self):
        self.model = _GoodBackbone()


class _ModelWithNoTrustedAttr:
    def __init__(self):
        self.backbone = _GoodBackbone()


class _ModelWithNonCallableTrustedAttr:
    def __init__(self):
        self.model = object()


def test_resolve_backbone_raises_when_missing_trusted_attr():
    model = MlxEmbeddingModel("dummy")
    model._model = _ModelWithNoTrustedAttr()
    with pytest.raises(RuntimeError, match="Could not resolve a trusted embedding backbone"):
        model._resolve_backbone()


def test_resolve_backbone_raises_when_trusted_attr_not_callable():
    model = MlxEmbeddingModel("dummy")
    model._model = _ModelWithNonCallableTrustedAttr()
    with pytest.raises(RuntimeError, match="is not callable"):
        model._resolve_backbone()


def test_run_backbone_accepts_valid_hidden_shape():
    model = MlxEmbeddingModel("dummy")
    model._model = _ModelWithTrustedAttr()
    hidden = model._run_backbone([[1, 2, 3]])
    assert hidden.shape == (1, 4, 8)


def test_run_backbone_rejects_invalid_hidden_shape():
    class _Model:
        def __init__(self):
            self.model = _BadBackbone()

    model = MlxEmbeddingModel("dummy")
    model._model = _Model()
    with pytest.raises(RuntimeError, match="Expected \\(batch, seq, dim\\)"):
        model._run_backbone([[1, 2, 3]])
