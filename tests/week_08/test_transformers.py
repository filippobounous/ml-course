"""Week 8 — attention / RoPE / BPE checks + torch-gated GPT shape checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "08_transformers"
    / "problems"
    / "solutions.py"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "08_tinygpt" / "model.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w8_solutions")


def test_softmax_row_sums_to_one(sols):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 7))
    p = sols.softmax(X, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), np.ones(4), atol=1e-12)


def test_softmax_shift_invariance(sols):
    x = np.array([1.0, 2.0, 3.0])
    a = sols.softmax(x)
    b = sols.softmax(x + 100.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_causal_attention_masks_future(sols):
    rng = np.random.default_rng(0)
    T, d = 5, 4
    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))
    _, weights = sols.scaled_dot_product_attention(Q, K, V, causal=True)
    # Above the diagonal must be zero.
    assert np.allclose(np.triu(weights, k=1), 0.0)
    # Rows must sum to 1.
    np.testing.assert_allclose(weights.sum(axis=-1), np.ones(T), atol=1e-12)


def test_multi_head_attention_output_shape(sols):
    rng = np.random.default_rng(0)
    T, d, heads = 6, 8, 4
    X = rng.standard_normal((T, d))
    W_Q = rng.standard_normal((d, d))
    W_K = rng.standard_normal((d, d))
    W_V = rng.standard_normal((d, d))
    W_O = rng.standard_normal((d, d))
    out = sols.multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads=heads, causal=True)
    assert out.shape == (T, d)


def test_sinusoidal_positional_encoding_shape(sols):
    pe = sols.sinusoidal_positional_encoding(T=10, d=8)
    assert pe.shape == (10, 8)
    # First column is sin(pos / 10000^0) = sin(pos).
    np.testing.assert_allclose(pe[:, 0], np.sin(np.arange(10)), atol=1e-12)


def test_rope_relative_position_invariance(sols):
    rng = np.random.default_rng(0)
    d = 8
    x = rng.standard_normal(d)
    y = rng.standard_normal(d)
    # <RoPE(x,3), RoPE(y,7)> == <RoPE(x,10), RoPE(y,14)> — same offset 4.
    a = sols.rope_inner_product(x, y, t=3, s=7)
    b = sols.rope_inner_product(x, y, t=10, s=14)
    assert abs(a - b) < 1e-10


def test_bpe_round_trip(sols):
    corpus = "the cat sat on the mat. the cat purrs. the mat is soft."
    merges = sols.train_bpe(corpus, vocab_size=400)
    # Merges should learn at least a handful of pairs.
    assert len(merges) > 3
    text = "the cat sat"
    ids = sols.bpe_encode(text, merges)
    assert sols.bpe_decode(ids, merges) == text


def test_bpe_learns_common_pairs_first(sols):
    corpus = "aaaaaaaaaaaaaa" + "bc" * 3
    merges = sols.train_bpe(corpus, vocab_size=260)
    # The first merge should be the most frequent pair; "aa" here.
    assert merges[0][:2] == (ord("a"), ord("a"))


# -- torch-gated GPT checks ----------------------------------------------------


@pytest.fixture(scope="module")
def model_module():
    pytest.importorskip("torch")
    return _load(MODEL_PATH, "w8_model")


def test_gpt_forward_shape(model_module):
    import torch

    cfg = model_module.GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, d_model=32)
    model = model_module.GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0


def test_gpt_weight_tying(model_module):
    import torch

    cfg = model_module.GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, d_model=16)
    model = model_module.GPT(cfg)
    assert model.head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    # Sanity: a gradient step updates both simultaneously.
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(x, x)
    loss.backward()
    assert torch.equal(model.head.weight.grad, model.token_embedding.weight.grad)


def test_gpt_generate_appends_tokens(model_module):
    import torch

    cfg = model_module.GPTConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2, d_model=16)
    model = model_module.GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=5)
    assert out.shape == (1, 4 + 7)
    # First 4 tokens unchanged.
    assert torch.equal(out[:, :4], prompt)
