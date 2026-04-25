# Week 7 — CNNs and vision

## Learning objectives

1. Derive the **convolution operator** and its gradient; understand receptive fields, strides, dilation, padding.
2. Build and train **ResNet-18** on CIFAR-10 (from scratch); compare to transfer learning with a pretrained ResNet / MobileNet backbone.
3. Produce **Grad-CAM** visualisations and a **failure-mode analysis** that goes beyond top-1 accuracy.
4. First look at the **Vision Transformer** (ViT) as a bridge to Week 8.

## Topics

- 2D convolution, shared weights, translation equivariance.
- Classic architectures: LeNet → AlexNet → VGG → ResNet → EfficientNet; structural comparison.
- Batch norm in CNNs; train/eval mode subtleties; running statistics.
- Transfer learning: feature extractor vs fine-tuning; the pretrained-on-ImageNet convention.
- Explainability: Grad-CAM, saliency, adversarial examples (FGSM).
- Preview: patch-based ViT.

## Deliverables

- Portfolio artifact: `portfolio/07_vision_classifier/` — CIFAR-10 classifier (ResNet-18 from scratch) + transfer-learning baseline + Grad-CAM + failure-mode analysis report.

## Reading plan

See `readings.md`.
