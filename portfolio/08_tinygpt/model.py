"""Decoder-only transformer (tiny GPT) — ~10M params, CPU / MPS friendly.

Pre-LN, GELU FFN, tied embedding+output, learned positional embeddings.
Deliberately written as a single readable file; no architecture tricks beyond
what the Week 8 notes describe.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "This module requires PyTorch. Install with `pip install -e '.[dl,ops]'`."
    ) from e


@dataclass
class GPTConfig:
    vocab_size: int = 10_000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1
    bias: bool = False  # GPT-2-style no-bias on linears except LM head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal masking.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        mask = self.causal_mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer with tied embeddings + learned positions."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying.
        self.head.weight = self.token_embedding.weight
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = idx.shape[1]
        if self.cfg.block_size < T:
            raise ValueError(f"Sequence length {T} exceeds block size {self.cfg.block_size}.")
        pos = torch.arange(T, device=idx.device)
        tok = self.token_embedding(idx)  # (B, T, C)
        ppe = self.position_embedding(pos)[None, :, :]  # (1, T, C)
        x = self.drop(tok + ppe)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-9)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
