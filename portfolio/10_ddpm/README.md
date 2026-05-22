# 10 — DDPM vs DDIM on FashionMNIST

Train a ~5M-parameter UNet-DDPM from scratch on FashionMNIST and ship an
honest DDPM-vs-DDIM step-count ablation.

## Layout

- `ddpm.py` — the UNet, timestep embeddings, linear noise schedule, DDPM
  loss, DDPM and DDIM samplers (eta≥0 supported).
- `train.py` — full training loop (10 epochs → plausible samples on MPS in
  ~2 hours). Saves `checkpoint.pt` and a `samples.png` grid.
- `fid.py` — **Fréchet Inception Distance** implementation: InceptionV3
  pool3 features (greyscale-repeated and bilinear-upsampled to
  $3 \times 299 \times 299$), Gaussian-fit per set, Heusel-2017
  closed-form FID. The FID math (`statistics`, `frechet_distance`) is
  pluggable — pass any feature extractor that returns a $(N, D)$ array.
- `ablate.py` — runs DDPM (1000) and DDIM η=0 at {10, 20, 50, 100} steps
  from the trained checkpoint; reports **both FID and the pixel-stat proxy**
  side-by-side, plus a sample grid. Use `--no-fid` to skip the
  InceptionV3 download for a quick smoke check.

## Reproduce

```bash
python -m pip install -e ".[dl,diffusion,ops]"

# 1) Train (Hydra entry point — see src/mlcourse/configs/week10/ddpm.yaml).
python portfolio/10_ddpm/train.py                          # defaults
python portfolio/10_ddpm/train.py quick=true               # CI smoke
python portfolio/10_ddpm/train.py trainer.max_epochs=20 diffusion.T=500

# 2) Ablate (FID + pixel-stat).
python portfolio/10_ddpm/ablate.py
python portfolio/10_ddpm/ablate.py --no-fid                # skip InceptionV3
```

First FID run downloads ~100 MB of pretrained InceptionV3 weights to the
torchvision cache; subsequent runs reuse them.

## Quality metrics

Two metrics, side-by-side:

- **FID** (Heusel 2017): Fréchet distance between Gaussians fit to
  InceptionV3 pool3 features of generated vs real samples. The
  industry-standard image-generation metric; numerical-sensitive
  matrix square root via `scipy.linalg.sqrtm`.
- **Pixel-stat distance**: $\|\mu_g - \mu_r\|_2 + \|\sigma_g - \sigma_r\|_2$
  over per-pixel mean and std. Cheap, no extra dependencies, tracks FID
  qualitatively on FashionMNIST — useful as a sanity check when you
  can't afford the Inception forward pass.

Both should rank the samplers in the same order; if they disagree,
trust FID and read the disagreement as a signal that the proxy is
hitting its limits on this image distribution.

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
- **FID math** (`test_fid.py`): identical-Gaussian distance = 0, symmetry,
  monotonicity in noise scale, channel-adapter accepts greyscale and
  rejects unsupported channel counts. A slow-marked test exercises the
  real InceptionV3 pipeline end-to-end.

## What I learned

*To be filled after running train + ablate end-to-end.*

## Model card

See [`model_card.md`](model_card.md) — Mitchell-2019 schema (intended use, metrics, training data, caveats).
