# Model card — CIFAR-10 ResNet-18 (from scratch)

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Name.** ResNet-18 from scratch on CIFAR-10.
- **Architecture.** Standard ResNet-18 with the CIFAR stem (3×3 conv,
  no maxpool) instead of the ImageNet stem. ~11.2 M parameters.
- **Framework.** PyTorch via `mlcourse.Trainer` (W6 harness).
- **Hyperparameters.** SGD + momentum 0.9 + weight decay 5e-4 +
  Nesterov; LR 0.01; 10 epochs at batch 128; deterministic seeding.

## Intended use

- **Primary.** Demonstrate the W6 Trainer + Grad-CAM explainability
  + FGSM adversarial sweep end-to-end on a non-trivial vision task.
- **Out-of-scope.** Deploying a CIFAR classifier in production.
  Even at the target accuracy this model has ~10% error and is
  fragile to FGSM perturbations as small as $\varepsilon = 4/255$.

## Metrics

- **Top-1 test accuracy** on CIFAR-10 test set (10,000 images).
- **FGSM robustness curve** at $\varepsilon \in \{0, 1, 2, 4, 8\}/255$.
- **Grad-CAM localisation maps** for 8 correctly and 8 incorrectly
  classified images.

## Training / evaluation data

- **CIFAR-10** ([Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html)).
  50 k train / 10 k test, 32×32 RGB, 10 classes, balanced. No PII.

## Quantitative analyses

| Metric | Target | Verified |
|---|---|---|
| Test acc (10 epochs, MPS) | ≥ 90% | ⏳ (aspirational; needs hardware run) |
| FGSM acc at $\varepsilon = 8/255$ | ≤ 25% | ⏳ |
| Training runtime (MPS) | ~30 min | ⏳ |

## Caveats

- Single-seed result. Std across seeds is typically 0.5–1.0 pp on
  CIFAR-10 ResNet-18 at this scale.
- The Grad-CAM panels are useful as **diagnostic**, not as evidence
  of "what the model learned" — that interpretation is contested
  (cf. Adebayo et al. 2018 on saliency-method sanity checks).
- FGSM is the easiest adversarial attack; a strong PGD attack will
  drop accuracy further. Don't claim adversarial robustness on
  FGSM-only numbers.
