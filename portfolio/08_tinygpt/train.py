"""Train a tiny GPT on TinyStories (or a local text file).

Runs on MPS / CPU. Target for MPS: ≤ 6 hours to coherent-text regime on the
default ~10M-param config; for CPU, plan for overnight or use `--max-iters`
to cap the run.

The BPE tokenizer is trained the first time `--tokenizer` doesn't exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_or_train_tokenizer(tokenizer_path: Path, corpus: str, vocab_size: int):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.trainers import BpeTrainer

    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[BOS]", "[EOS]"], show_progress=False
    )
    tok.train_from_iterator([corpus], trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(tokenizer_path))
    return tok


def _load_corpus(corpus_path: Path) -> str:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. "
            "Download TinyStories with `huggingface-cli download roneneldan/TinyStories` "
            "and concatenate the splits into one text file."
        )
    return corpus_path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=str(HERE / "data" / "tinystories.txt"))
    parser.add_argument("--tokenizer", type=str, default=str(HERE / "tokenizer.json"))
    parser.add_argument("--out", type=str, default=str(HERE / "checkpoint.pt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=4000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    args = parser.parse_args()

    import numpy as np
    import torch
    from model import GPT, GPTConfig, count_parameters

    from mlcourse.utils import detect_device, seed_everything

    seed_everything(0)
    device = detect_device()
    print(f"device: {device}")

    corpus = _load_corpus(Path(args.corpus))
    tok = _load_or_train_tokenizer(Path(args.tokenizer), corpus, args.vocab_size)
    ids = np.array(tok.encode(corpus).ids, dtype=np.int64)
    split = int(0.9 * len(ids))
    train_ids = torch.from_numpy(ids[:split])
    val_ids = torch.from_numpy(ids[split:])

    cfg = GPTConfig(
        vocab_size=tok.get_vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
    )
    model = GPT(cfg).to(device)
    print(f"params: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    def get_batch(split_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(split_tensor) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([split_tensor[i : i + cfg.block_size] for i in ix])
        y = torch.stack([split_tensor[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss() -> tuple[float, float]:
        model.eval()
        train_losses = [
            float(model(*get_batch(train_ids))[1] or torch.tensor(0.0)) for _ in range(20)
        ]
        val_losses = [float(model(*get_batch(val_ids))[1] or torch.tensor(0.0)) for _ in range(20)]
        model.train()
        return sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, args.max_iters + 1):
        x, y = get_batch(train_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            tl, vl = estimate_loss()
            dt = time.perf_counter() - t0
            print(f"  step {step:5d}  train={tl:.4f}  val={vl:.4f}  {dt:.1f}s")
            history.append({"step": step, "train": tl, "val": vl})
            t0 = time.perf_counter()

    # Save a checkpoint.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": cfg.__dict__, "history": history},
        args.out,
    )
    print(f"saved checkpoint: {args.out}")

    # Generate a sample.
    prompt = "Once upon a time,"
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=50)
    print("\nsample:\n" + tok.decode(out[0].cpu().tolist()))
    (HERE / "samples.md").write_text(
        "# Generated samples\n\n```\n" + tok.decode(out[0].cpu().tolist()) + "\n```\n",
        encoding="utf-8",
    )
    (HERE / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
