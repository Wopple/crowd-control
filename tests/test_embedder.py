"""Tests for embedding protocol and providers."""

import os
from unittest.mock import patch

import pytest
from conftest import FakeEmbedder

from crowd_control.config import EmbeddingConfig
from crowd_control.embed.base import create_embedder


class TestFakeEmbedder:
    def test_dimensions(self):
        embedder = FakeEmbedder(dimensions=16)
        vectors = embedder.embed(["hello"])
        assert len(vectors[0]) == 16
        assert embedder.dimensions == 16

    def test_deterministic(self):
        embedder = FakeEmbedder()
        v1 = embedder.embed(["same text"])[0]
        v2 = embedder.embed(["same text"])[0]
        assert v1 == v2

    def test_distinct(self):
        embedder = FakeEmbedder()
        v1 = embedder.embed(["text one"])[0]
        v2 = embedder.embed(["text two"])[0]
        assert v1 != v2

    def test_batch(self):
        embedder = FakeEmbedder()
        vectors = embedder.embed(["a", "b", "c"])
        assert len(vectors) == 3

    def test_empty_list(self):
        embedder = FakeEmbedder()
        assert embedder.embed([]) == []

    def test_normalized(self):
        embedder = FakeEmbedder()
        vector = embedder.embed(["test"])[0]
        norm = sum(x**2 for x in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-6


class TestCreateEmbedder:
    def test_unknown_provider(self):
        config = EmbeddingConfig(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder(config)


class TestVoyageMissingKey:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove VOYAGE_API_KEY if present
            os.environ.pop("VOYAGE_API_KEY", None)
            try:
                from crowd_control.embed.voyage import VoyageEmbedder

                with pytest.raises(ValueError, match="Voyage API key not found"):
                    VoyageEmbedder()
            except ImportError:
                pytest.skip("voyageai not installed")


class TestOpenAIMissingKey:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                from crowd_control.embed.openai import OpenAIEmbedder

                with pytest.raises(ValueError, match="OpenAI API key not found"):
                    OpenAIEmbedder()
            except ImportError:
                pytest.skip("openai not installed")
