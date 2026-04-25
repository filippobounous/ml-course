# Week 7 — CNNs and vision (lecture notes)

*Reading pair: Goodfellow DL Ch.9 · He *ResNet* 2015 · Selvaraju *Grad-CAM* 2017 · Dosovitskiy *ViT* 2020 (first pass).*

---

## 1. Why convolutions

Images have two structural priors fully-connected nets don't use:

1. **Locality.** Meaningful patterns (edges, textures) span a small neighbourhood of pixels.
2. **Translation equivariance.** A cat is still a cat shifted ten pixels right.

Convolutional layers bake both in: shared kernels applied at every spatial position. A `C_in × C_out × k × k` convolution has $C_\text{in} C_\text{out} k^2$ parameters versus $H W C_\text{in} C_\text{out}$ for a fully-connected layer — orders-of-magnitude fewer for real image sizes.

## 2. The 2-D convolution operator

For input $x \in \mathbb{R}^{C_\text{in} \times H \times W}$ and kernel $W \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}$,

$$y_{c_\text{out}, i, j} = \sum_{c_\text{in}=0}^{C_\text{in}-1} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W_{c_\text{out}, c_\text{in}, u, v} \, x_{c_\text{in}, i+u, j+v} + b_{c_\text{out}}.$$

Strictly speaking this is cross-correlation, not mathematical convolution, but "convolution" is the standard ML label. Same thing up to flipping the kernel.

**Receptive field.** Each output location is a function of a $k \times k$ input patch. Stack two 3×3 convs and the receptive field grows to 5×5. With dilations, strides, and pooling, a modern CNN stack has receptive fields covering most of the input.

**Strides and padding.** Stride $s$ downsamples output spatial size by $s$. "Same" padding (`floor(k/2)`) preserves spatial dimensions for stride 1; "valid" padding drops the border.

## 3. Gradient of a conv layer

The gradient $\partial L / \partial W$ is itself a convolution:

- $\partial L / \partial W_{c_\text{out}, c_\text{in}, u, v} = \sum_{i, j} (\partial L / \partial y_{c_\text{out}, i, j}) \cdot x_{c_\text{in}, i+u, j+v}$.

The gradient $\partial L / \partial x$ is a convolution with the *flipped* kernel (sometimes called a "transposed convolution"):

- $\partial L / \partial x_{c_\text{in}, i, j} = \sum_{c_\text{out}, u, v} W_{c_\text{out}, c_\text{in}, u, v} \cdot (\partial L / \partial y_{c_\text{out}, i-u, j-v})$.

Instructive to derive once, then trust PyTorch.

## 4. Normalisation and the train / eval split

**BatchNorm** per channel: $\hat x = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$, then $\gamma \hat x + \beta$. Running statistics $\mu_\text{r}, \sigma_\text{r}^2$ are updated with momentum during training; at eval time those are used instead of the batch statistics. This **must** be correct or your eval numbers are wrong.

Pitfalls:
- `model.eval()` switches BN to running-stats mode. Forget it and you get unstable small-batch eval.
- Small batch sizes (< ~16) destabilise BN during training; consider **GroupNorm** or **LayerNorm** alternatives.
- Distributed training needs `SyncBatchNorm` across replicas.

## 5. ResNet: residual connections

A plain deep stack is hard to optimise: gradients vanish, accuracy saturates then degrades. Residual blocks re-express a layer as $y = F(x) + x$, which means the identity mapping has zero loss — a strong inductive bias. The optimiser only needs to learn the residual $F$.

ResNet-18 block (two 3×3 convs with BN + ReLU, plus an identity or 1×1 skip) is the canonical building block. The deeper cousins ResNet-34/50/101/152 differ mostly in block count and in the use of 1×1 bottlenecks (ResNet-50+).

For CPU / Apple Silicon training, **ResNet-18 on CIFAR-10** is the sweet spot: 10–40 minutes for 10 epochs on M-series, ~90% test accuracy.

## 6. Transfer learning

Never train on ImageNet from scratch on a laptop. Instead:

- **Feature extraction.** Freeze a pretrained backbone (ResNet-18, MobileNetV3-Small, EfficientNet-B0 from `torchvision.models`), replace the classification head, train only the head. Seconds per epoch on CPU for small datasets.
- **Fine-tuning.** Unfreeze the last block or two, lower the learning rate by 10× on the backbone. Often gives a 1–3% bump over feature extraction at the cost of a bit more compute.

Always preprocess the test images with the same normalisation constants the backbone was trained with — the `torchvision.models` docs list them.

## 7. Explainability: Grad-CAM

Grad-CAM (Selvaraju 2017) visualises which input regions drove a class score.

1. Pick a target conv layer (the last block, typically).
2. Compute $y^c$, the logit for the class of interest.
3. Pool the gradients $\partial y^c / \partial A^k$ per feature-map channel $A^k$ → channel weights $\alpha_k^c$.
4. The class-discriminative localisation map is $L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$, upsampled to input resolution.

Used well it tells you *why* the model is right or wrong on specific examples — a much better diagnostic than aggregate accuracy.

## 8. Adversarial fragility

Ten years after AlexNet we still train networks that misclassify images with imperceptible perturbations.

**FGSM** (Goodfellow et al. 2014). Given image $x$, label $y$, loss $L$,

$$x' = x + \varepsilon \cdot \operatorname{sign}(\nabla_x L(\theta, x, y)).$$

Sweep $\varepsilon$ in steps of $1/255$ and plot accuracy. A network at 90% clean drops to ~20% at $\varepsilon = 8/255$ — sobering.

## 9. Vision transformers (preview)

ViT cuts an image into 16×16 patches, embeds each patch, adds positional encodings, and feeds a standard transformer encoder. On ImageNet-scale data ViTs match or beat CNNs; on smaller datasets CNNs' inductive biases usually still help. We build transformers from scratch next week.

## What to do with these notes

Work the problem set in `../problems/README.md`. Implement manual 2-D convolution in NumPy (reference in `../problems/solutions.py`). Build the portfolio artifact in `../../../portfolio/07_vision_classifier/`: CIFAR-10 ResNet-18 + transfer-learning baseline + Grad-CAM + FGSM.
