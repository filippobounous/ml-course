# Problem set — Week 7

## Theory

1. **Conv backprop.** Derive $\partial L / \partial W$ for a 2D convolution layer. Show its structure is itself a convolution.
2. **ResNet-18 parameter count.** Derive the exact parameter count of a stock torchvision ResNet-18. Verify with `torchinfo`.
3. **Receptive field.** Compute the receptive field at each stage of ResNet-18 in closed form.

## Implementation

4. Train **ResNet-18 from scratch on CIFAR-10** using your Week 6 `Trainer`. Aim for ≥90% test accuracy in under 30 minutes on MPS.
5. **Transfer-learning baseline**: freeze a torchvision pretrained ResNet-18 or MobileNetV3-Small backbone and fine-tune a classification head on CIFAR-10. Compare to #4 on accuracy, training time, and sample efficiency.
6. **Grad-CAM visualisations.** Visualise the class activations for 10 correctly and 10 incorrectly classified images.

## Applied (portfolio)

7. **Failure-mode analysis** (in `portfolio/07_vision_classifier/`): pick the 50 hardest test images (highest loss), tag them by failure type (blur, occlusion, wrong class per ImageNet prior, adversarial-prone), and write a short analysis. Include an **FGSM adversarial** example showing perturbation magnitude vs accuracy drop.

## Grading

Tests in `tests/week_07/` check that (a) the classifier hits a threshold accuracy on a held-out subset and (b) Grad-CAM outputs have the right spatial shape.
