# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coreason_search.config import EmbeddingConfig
from coreason_search.embedder import MockEmbedder, get_embedder, reset_embedder
from coreason_search.embedders.hf import HuggingFaceEmbedder


@pytest.fixture  # type: ignore[misc]
def clean_embedder() -> Generator[None, None, None]:
    reset_embedder()
    yield
    reset_embedder()


def test_get_embedder_defaults_to_auto_fallback(clean_embedder: None) -> None:
    """Test that default 'auto' falls back to MockEmbedder if libs missing."""
    # We assume 'sentence_transformers' is NOT installed in this env
    # But just in case, we patch it out
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        config = EmbeddingConfig(provider="auto")
        embedder = get_embedder(config)
        assert isinstance(embedder, MockEmbedder)


def test_get_embedder_explicit_mock(clean_embedder: None) -> None:
    config = EmbeddingConfig(provider="mock")
    embedder = get_embedder(config)
    assert isinstance(embedder, MockEmbedder)


def test_get_embedder_explicit_hf_missing_dep(clean_embedder: None) -> None:
    """Explicit 'hf' should raise ImportError if deps missing."""
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        config = EmbeddingConfig(provider="hf")
        with pytest.raises(ImportError, match="sentence-transformers is not installed"):
            get_embedder(config)


def test_hf_embedder_init_success(clean_embedder: None) -> None:
    """Test successful initialization of HuggingFaceEmbedder with mocked SentenceTransformer."""
    mock_st_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_cls.return_value = mock_model_instance

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="hf", model_name="test-model", context_length=512)
            embedder = get_embedder(config)

            assert isinstance(embedder, HuggingFaceEmbedder)
            mock_st_cls.assert_called_once_with("test-model", trust_remote_code=True)
            assert mock_model_instance.max_seq_length == 512


def test_hf_embedder_embed(clean_embedder: None) -> None:
    """Test embed method of HuggingFaceEmbedder."""
    mock_st_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_cls.return_value = mock_model_instance

    # Setup mock return for encode
    expected_emb = np.array([[0.1, 0.2]], dtype=np.float32)
    mock_model_instance.encode.return_value = expected_emb

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="hf")
            embedder = get_embedder(config)

            # Test single string
            res = embedder.embed("hello")
            assert np.array_equal(res, expected_emb)
            mock_model_instance.encode.assert_called_with(
                ["hello"], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )

            # Test list
            embedder.embed(["a", "b"])
            mock_model_instance.encode.assert_called_with(
                ["a", "b"], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )


def test_hf_embedder_empty_input(clean_embedder: None) -> None:
    """Test embed method with empty input."""
    mock_st_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_cls.return_value = mock_model_instance
    mock_model_instance.get_sentence_embedding_dimension.return_value = 768

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="hf")
            embedder = get_embedder(config)

            res = embedder.embed([])
            assert res.shape == (0, 768)
            mock_model_instance.encode.assert_not_called()


def test_auto_fallback_on_init_error(clean_embedder: None) -> None:
    """Test that 'auto' falls back to MockEmbedder if model init crashes."""
    mock_st_cls = MagicMock()
    mock_st_cls.side_effect = RuntimeError("Model not found")

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="auto")
            embedder = get_embedder(config)

            # Should catch RuntimeError and return Mock
            assert isinstance(embedder, MockEmbedder)


def test_explicit_hf_init_error(clean_embedder: None) -> None:
    """Test that explicit 'hf' raises error if model init crashes."""
    mock_st_cls = MagicMock()
    mock_st_cls.side_effect = RuntimeError("Model not found")

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="hf")

            # Should propagate RuntimeError
            with pytest.raises(RuntimeError, match="Model not found"):
                get_embedder(config)


def test_embed_mixed_input(clean_embedder: None) -> None:
    """Test embedding mixed valid and empty strings/unicode."""
    mock_st_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_cls.return_value = mock_model_instance

    expected_emb = np.array([[0.1], [0.0], [0.5]], dtype=np.float32)
    mock_model_instance.encode.return_value = expected_emb

    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        with patch("sentence_transformers.SentenceTransformer", mock_st_cls):
            config = EmbeddingConfig(provider="hf")
            embedder = get_embedder(config)

            mixed_input = ["valid", "", "🚀"]
            res = embedder.embed(mixed_input)

            assert np.array_equal(res, expected_emb)
            mock_model_instance.encode.assert_called_with(
                mixed_input, batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
