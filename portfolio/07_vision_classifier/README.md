# 07 — CIFAR-10 classifier + failure analysis

Train a **ResNet-18 from scratch** on CIFAR-10, run **Grad-CAM** on the last
conv block, and produce an **FGSM adversarial-robustness curve**.

## What's here

- `classifier.py` — `get_cifar10_loaders()`, `resnet18_for_cifar()`
  (3×3 stem + no maxpool, standard CIFAR recipe), `transfer_resnet18()`
  (ImageNet-pretrained feature extractor), `GradCAM`, `fgsm()`.
- `demo.py` — end-to-end training **via `mlcourse.Trainer`** + Grad-CAM +
  FGSM sweep. Runs in ~30 min on MPS, ~3 h on CPU. Use `quick=true` for a
  one-epoch smoke check. This is the first downstream week that consumes
  the Week-6 `Trainer` harness; `tests/week_07/test_slow_trainer_integration.py`
  guards the integration. Driven by Hydra: see
  `src/mlcourse/configs/week07/cifar10.yaml`.

## Reproduce

```bash
python -m pip install -e ".[dl,ops]"
python portfolio/07_vision_classifier/demo.py              # defaults
python portfolio/07_vision_classifier/demo.py quick=true   # CI smoke
python portfolio/07_vision_classifier/demo.py trainer.max_epochs=20 trainer.lr=0.05
```

First run downloads CIFAR-10 to `portfolio/07_vision_classifier/data/` (170 MB).

## Outputs

- `report.md` — final accuracy and FGSM curve table.
- `gradcam.png` — 2-row panel: eight test images with their Grad-CAM overlays.

## Target accuracy

- ResNet-18 from scratch, 10 epochs on MPS: ≥ 90% test accuracy.
- Transfer-learning baseline (frozen ImageNet backbone, linear head):
  ~70–75% on CIFAR-10 (the domain gap shows). Train `transfer_resnet18(freeze_backbone=False)`
  with a low LR on the backbone and you recover most of the gap.

## Tests

`tests/week_07/` covers the manual NumPy conv (`modules/07_cnns_vision/problems/solutions.py`)
plus torch-gated shape checks on Grad-CAM and FGSM.

## What I learned

*To be filled after running the demo end-to-end. Suggested bullets:*

- How the 3×3/no-maxpool CIFAR stem preserves spatial resolution that the
  standard ImageNet recipe throws away.
- How Grad-CAM's localisation flips from plausible to nonsense on the hardest
  test images — a much better diagnostic than aggregate accuracy.
- How quickly FGSM erodes accuracy — 90% → 20% at ε=8/255 is typical for a
  non-adversarially-trained model.
