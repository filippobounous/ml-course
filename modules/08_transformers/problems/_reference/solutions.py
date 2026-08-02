"""Week 8 — reference solutions.

Pure-NumPy helpers for the attention derivations. Torch-based transformer
code lives in `portfolio/08_tinygpt/`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Scaled dot-product attention (NumPy reference)


def softmax(x: ArrayF, *, axis: int = -1) -> ArrayF:
    """Numerically stable softmax with the max-shift trick."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: ArrayF, K: ArrayF, V: ArrayF, *, causal: bool = False
) -> tuple[ArrayF, ArrayF]:
    """Return (output, attention_weights).

    Q, K: (T, d_k)
    V:    (T, d_v)
    """
    d_k = Q.shape[-1]
    logits = Q @ K.T / np.sqrt(d_k)
    if causal:
        T = logits.shape[0]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        logits = np.where(mask, -np.inf, logits)
    weights = softmax(logits, axis=-1)
    return weights @ V, weights


def multi_head_attention(
    X: ArrayF,
    W_Q: ArrayF,
    W_K: ArrayF,
    W_V: ArrayF,
    W_O: ArrayF,
    *,
    n_heads: int,
    causal: bool = False,
) -> ArrayF:
    """Multi-head attention computed by concatenating per-head outputs.

    X:   (T, d)
    W_*: (d, d)  — single combined projections
    W_O: (d, d)  — output projection
    Heads split the hidden dim evenly: d_head = d // n_heads.
    """
    d = X.shape[-1]
    if d % n_heads != 0:
        raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
    d_head = d // n_heads
    Q = X @ W_Q  # (T, d)
    K = X @ W_K
    V = X @ W_V

    heads = []
    for h in range(n_heads):
        sl = slice(h * d_head, (h + 1) * d_head)
        out_h, _ = scaled_dot_product_attention(Q[:, sl], K[:, sl], V[:, sl], causal=causal)
        heads.append(out_h)
    concat = np.concatenate(heads, axis=-1)  # (T, d)
    return concat @ W_O


# -----------------------------------------------------------------------------
# Sinusoidal positional encodings (Vaswani 2017)


def sinusoidal_positional_encoding(T: int, d: int) -> ArrayF:
    if d % 2 != 0:
        raise ValueError("d must be even for the sin/cos pairing.")
    pe = np.zeros((T, d), dtype=np.float64)
    positions = np.arange(T)[:, None]
    div_term = np.exp(-np.log(10000.0) * np.arange(0, d, 2) / d)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


# -----------------------------------------------------------------------------
# RoPE rotation — relative-position check


def apply_rope(x: ArrayF, positions: ArrayF) -> ArrayF:
    """Apply rotary positional embedding to a (T, d) array with even d.

    Pairs adjacent dimensions (2i, 2i+1) and rotates each pair by an angle
    θ_{t, i} = t / 10000^{2i/d}.
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError("d must be even for RoPE.")
    half = d // 2
    freqs = 1.0 / (10000.0 ** (np.arange(half) / half))
    angles = positions[:, None] * freqs[None, :]  # (T, d/2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def rope_inner_product(x: ArrayF, y: ArrayF, t: int, s: int) -> float:
    """<RoPE(x, t), RoPE(y, s)> — depends only on t − s under RoPE."""
    xr = apply_rope(x[None, :], np.array([t]))[0]
    yr = apply_rope(y[None, :], np.array([s]))[0]
    return float(xr @ yr)


# -----------------------------------------------------------------------------
# Byte-level BPE (pedagogical implementation)


def train_bpe(corpus: str, *, vocab_size: int) -> list[tuple[int, int, int]]:
    """Train a byte-level BPE on `corpus`.

    Returns a list of `(pair_left, pair_right, new_token)` merges in the order
    they were learned. `pair_*` are byte values (0–255) or previously-merged
    token ids; `new_token` is the id assigned to the merged pair.

    This is an instructive reference implementation only — for real training
    use the `tokenizers` library.
    """
    if vocab_size <= 256:
        return []

    # Treat the corpus as a list of byte tokens.
    tokens: list[int] = list(corpus.encode("utf-8"))
    merges: list[tuple[int, int, int]] = []
    next_id = 256

    from itertools import pairwise

    while next_id < vocab_size:
        counts: dict[tuple[int, int], int] = {}
        for a, b in pairwise(tokens):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        pair = max(counts, key=counts.get)  # type: ignore[arg-type]
        if counts[pair] < 2:
            break

        # Apply the merge.
        new_tokens: list[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(next_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        merges.append((pair[0], pair[1], next_id))
        next_id += 1

    return merges


def bpe_encode(text: str, merges: list[tuple[int, int, int]]) -> list[int]:
    """Apply learned BPE merges to `text`, producing token ids."""
    tokens: list[int] = list(text.encode("utf-8"))
    for a, b, new_id in merges:
        out: list[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def bpe_decode(ids: list[int], merges: list[tuple[int, int, int]]) -> str:
    """Inverse of `bpe_encode`: expand merged ids back to bytes and decode UTF-8."""
    # Map: merged_id -> (left, right) so we can recursively split.
    inverse = {new_id: (a, b) for a, b, new_id in merges}

    def expand(i: int) -> list[int]:
        if i < 256:
            return [i]
        a, b = inverse[i]
        return expand(a) + expand(b)

    bytes_out = bytearray()
    for i in ids:
        bytes_out.extend(expand(i))
    return bytes_out.decode("utf-8", errors="replace")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 8))
    W_Q = rng.standard_normal((8, 8))
    W_K = rng.standard_normal((8, 8))
    W_V = rng.standard_normal((8, 8))
    W_O = rng.standard_normal((8, 8))
    out = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads=2, causal=True)
    print("MHA out:", out.shape)

    merges = train_bpe("the cat sat on the mat the cat purrs", vocab_size=280)
    encoded = bpe_encode("the cat", merges)
    decoded = bpe_decode(encoded, merges)
    print("BPE round-trip:", decoded)
