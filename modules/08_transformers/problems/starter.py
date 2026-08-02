"""Week 08 — problem-set starter. **This is the file you edit.**

`pytest tests/week_08` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_08
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Scaled dot-product attention (NumPy reference)


def softmax(x: ArrayF, *, axis: int = -1) -> ArrayF:
    """Numerically stable softmax with the max-shift trick."""
    raise NotImplementedError("softmax")


def scaled_dot_product_attention(
    Q: ArrayF, K: ArrayF, V: ArrayF, *, causal: bool = False
) -> tuple[ArrayF, ArrayF]:
    """Return (output, attention_weights).

    Q, K: (T, d_k)
    V:    (T, d_v)
    """
    raise NotImplementedError("scaled_dot_product_attention")


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
    raise NotImplementedError("multi_head_attention")


# -----------------------------------------------------------------------------
# Sinusoidal positional encodings (Vaswani 2017)


def sinusoidal_positional_encoding(T: int, d: int) -> ArrayF:
    raise NotImplementedError("sinusoidal_positional_encoding")


# -----------------------------------------------------------------------------
# RoPE rotation — relative-position check


def apply_rope(x: ArrayF, positions: ArrayF) -> ArrayF:
    """Apply rotary positional embedding to a (T, d) array with even d.

    Pairs adjacent dimensions (2i, 2i+1) and rotates each pair by an angle
    θ_{t, i} = t / 10000^{2i/d}.
    """
    raise NotImplementedError("apply_rope")


def rope_inner_product(x: ArrayF, y: ArrayF, t: int, s: int) -> float:
    """<RoPE(x, t), RoPE(y, s)> — depends only on t − s under RoPE."""
    raise NotImplementedError("rope_inner_product")


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
    raise NotImplementedError("train_bpe")


def bpe_encode(text: str, merges: list[tuple[int, int, int]]) -> list[int]:
    """Apply learned BPE merges to `text`, producing token ids."""
    raise NotImplementedError("bpe_encode")


def bpe_decode(ids: list[int], merges: list[tuple[int, int, int]]) -> str:
    """Inverse of `bpe_encode`: expand merged ids back to bytes and decode UTF-8."""
    raise NotImplementedError("bpe_decode")
