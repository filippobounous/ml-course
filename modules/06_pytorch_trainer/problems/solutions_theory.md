# Week 6 — Theory-problem solutions

## 1. Autograd bookkeeping

- **`requires_grad`** is a property of leaf tensors. Setting it to `True` tells PyTorch to build a computational graph containing operations on that tensor. Intermediate (non-leaf) tensors inherit `requires_grad=True` if any input did.
- **Leaf tensors** are tensors that the user constructed (via `torch.tensor`, `torch.randn`, `nn.Parameter`, etc.) rather than ones produced by an operation. Only leaves accumulate `.grad` under standard autograd (non-leaf tensors can be promoted via `.retain_grad()`).
- **`.detach()`** returns a new tensor sharing storage with the original but *not* in the computational graph. `.clone()` copies storage but preserves the graph. Use `.detach().clone()` when you want both a new memory region and a graph cut — the common case for saving tensors to return to the caller.
- **`.backward()` with `retain_graph=True`** keeps the forward-pass intermediates alive after the backward pass so you can call `.backward()` again later (needed if you want a second gradient w.r.t. different outputs). Without it, PyTorch frees the intermediates as backward unwinds the graph.

## 2. Mixed precision on MPS

- **Support matrix.** M1 / M2 chips support fp16 natively but not bf16 — bf16 operations fall back to fp32 or error. M3 and later also support bf16. Always check with `torch.backends.mps.is_bf16_supported()` before choosing dtype.
- **What `torch.autocast("mps")` casts.** It wraps a subset of ops — matmul, convolutions, some reductions — and runs them in the declared dtype (fp16 by default) while leaving unsafe-to-cast ops (softmax, layer-norm, loss functions, most reductions with accumulation) in fp32. This gives you the memory / throughput benefit on the big ops without the numerical instability that would follow from casting everything.
- **Practical upshot.** On M-series, fp16 autocast is typically 1.3–2× faster than pure fp32 for convnets and transformers. For LLM inference, native fp16 (no autocast) often wins because every op is in fp16 and the MPS kernels are optimised.

Links worth reading in the `torch.amp` docs: the op-list (what gets auto-cast, what doesn't) and the known-issues section on MPS.
