# 10 — DDPM vs DDIM on FashionMNIST

Train a ~5M-parameter UNet-DDPM from scratch on FashionMNIST and ship an
honest DDPM-vs-DDIM step-count ablation.

## Layout

- `ddpm.py` — the UNet, timestep embeddings, linear noise schedule, DDPM
  loss, DDPM and DDIM samplers (eta≥0 supported).
- `train.py` — full training loop (10 epochs → plausible samples on MPS in
  ~2 hours). Saves `checkpoint.pt` and a `samples.png` grid.
- `ablate.py` — runs DDPM (1000) and DDIM η=0 at {10, 20, 50, 100} steps
  from the trained checkpoint; reports a proxy quality metric
  (pixel-statistics distance to the FashionMNIST test set) and a sample
  grid across samplers.

## Reproduce

```bash
python -m pip install -e ".[dl,diffusion,ops]"

# 1) Train.
python portfolio/10_ddpm/train.py             # full run
python portfolio/10_ddpm/train.py --quick     # CI smoke

# 2) Ablate.
python portfolio/10_ddpm/ablate.py
```

## Note on the quality metric

Full FID needs an InceptionV3 checkpoint and pipeline that are overkill for
a coursework artifact. Pixel-statistics distance (L2 on mean + L2 on std,
per pixel) tracks FID qualitatively on FashionMNIST at the sample counts
we have compute for — and is trivial to audit.

Adding FID via `pytorch-fid` is a one-line swap once the baseline works.

## Expected ablation behaviour

At 10 epochs of training:
- DDPM (1000): baseline quality.
- DDIM η=0 at 100 steps: ~indistinguishable from DDPM.
- DDIM η=0 at 50 steps: minor quality drop.
- DDIM η=0 at 20 steps: visible loss of fine detail.
- DDIM η=0 at 10 steps: noticeable artefacts.

## Multimodal companion — CLIP retrieval

`open_clip` ViT-B/32 gives zero-shot retrieval out of the box. Suggested
notebook under `notebooks/`:

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tok = open_clip.get_tokenizer("ViT-B-32")
# Embed a folder of images + a set of natural-language queries, compute
# cosine similarities, retrieve top-k.
```

## Tests

`tests/week_10/` covers:
- Linear / cosine schedule shapes and boundary conditions.
- Closed-form q(x_t|x_0) stats.
- DDIM deterministic behaviour (two runs with same seed → identical samples).
- InfoNCE loss (identity embeddings recover the theoretical minimum).
- Torch-gated UNet forward pass + DDPM loss shape check.

## What I learned

*To be filled after running train + ablate end-to-end.*
