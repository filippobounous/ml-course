# 07 — CIFAR-10 classifier + failure-mode analysis

> Populated in Week 7. See `modules/07_cnns_vision/`.

## Problem
Train ResNet-18 from scratch on CIFAR-10 and compare to a transfer-learning
baseline; then dig into the failure modes with Grad-CAM and an FGSM
adversarial study.

## Method
- ResNet-18 from scratch (from `torchvision.models` but random init) on CIFAR-10.
- Transfer-learning baseline (frozen pretrained backbone + linear head).
- Grad-CAM on successes and failures.
- FGSM adversarial-perturbation sweep.

## Results
*Accuracy table (scratch vs transfer); Grad-CAM panels; FGSM curve.*

## Reproduce
```bash
make -C portfolio/07_vision_classifier reproduce
```
