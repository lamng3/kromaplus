import torch
import pytest
from kromaplus.embeddings.text_embedding import TextEmbedding


@pytest.fixture(scope="module")
def embedder():
    # singleton for all tests in this module
    return TextEmbedding()

def test_to_embedding_returns_tensor(embedder):
    emb = embedder.to_embedding("Hello, world!")
    # should be a 1‑D float tensor
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1
    assert emb.dtype == torch.float32 or emb.dtype == torch.float64

def test_embedding_has_nonzero_length(embedder):
    emb = embedder.to_embedding("Quick test")
    # dimensionality should be > 0
    assert emb.numel() > 0

def test_no_nans_or_infs(embedder):
    emb = embedder.to_embedding("Check for NaNs and Infs")
    assert not torch.isnan(emb).any(), "Embedding contains NaNs"
    assert not torch.isinf(emb).any(), "Embedding contains Infs"

def test_repeatability(embedder):
    text = "Repeatability"
    e1 = embedder.to_embedding(text)
    e2 = embedder.to_embedding(text)
    # identical calls should produce (numerically) the same output
    assert torch.allclose(e1, e2, atol=1e-6), "Embeddings differ between runs"

def test_different_texts_produce_different_embeddings(embedder):
    e1 = embedder.to_embedding("First sentence")
    e2 = embedder.to_embedding("A completely different sentence")
    # they shouldn’t be exactly equal
    assert not torch.allclose(e1, e2, atol=1e-6), "Different inputs gave identical embeddings"