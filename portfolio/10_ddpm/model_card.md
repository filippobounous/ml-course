# Model card — UNet-DDPM on FashionMNIST

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Name.** SmallUNet-DDPM on FashionMNIST.
- **Architecture.** UNet with timestep + class-label embeddings, ~5M
  parameters. Linear $\beta$-schedule, $T = 1000$ steps.
  Classifier-free guidance: 10% unconditional drop at training, $w$
  scale at sampling.
- **Framework.** PyTorch via `mlcourse.Trainer`'s `loss_fn=None`
  custom-loss path (W6 + the W4 Trainer-integration PR).
- **Hyperparameters.** Adam LR 2e-4, batch 128, 10 epochs, $T=1000$,
  EMA decay 0.999.

## Intended use

- **Primary.** Demonstrate end-to-end diffusion modelling: noise
  schedule, $\epsilon$-prediction loss, DDIM acceleration, CFG, and
  FID evaluation.
- **Out-of-scope.** Generating realistic images of anything other
  than FashionMNIST clothing items. The model has never seen RGB
  or anything larger than $28 \times 28$.

## Metrics

- **Fréchet Inception Distance (FID)** — `portfolio/10_ddpm/fid.py`
  with InceptionV3 pool3 features. Lower is better.
- **Pixel-statistics distance** — cheap proxy retained as cross-
  reference; agrees with FID on most ranking decisions.
- **DDIM vs DDPM step-count ablation** — sample quality at
  $\{1000, 100, 50, 20, 10\}$ steps.

## Training / evaluation data

- **FashionMNIST** (Xiao et al. 2017). 60 k train / 10 k test,
  28×28 greyscale, 10 classes, balanced. No PII.

## Quantitative analyses

| Metric | Target | Verified |
|---|---|---|
| FID at DDPM-1000 | < 20 | ⏳ (aspirational; needs hardware run) |
| FID at DDIM-50 ($\eta = 0$) | within 10% of DDPM-1000 | ⏳ |
| Training runtime (MPS) | ~2 h | ⏳ |

## Caveats

- FID is **dataset-relative** — a "FID of 15" on FashionMNIST is
  not comparable to "FID of 15" on CIFAR-10 or CelebA. Don't quote
  the number without the dataset.
- InceptionV3 was trained on ImageNet; using it as a feature
  extractor for FashionMNIST images mildly violates the assumption
  that the features are well-calibrated for the data distribution.
  This is a standard caveat for FID on non-ImageNet datasets.
- 256 generated samples is a small set; FID has high variance at
  that scale (Heusel 2017 recommends ≥ 10k). Headline numbers
  should be computed at 10k+ samples in any real comparison.
