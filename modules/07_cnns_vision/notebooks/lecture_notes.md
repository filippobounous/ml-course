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

Work the problem set in `../problems/README.md`. Implement manual 2-D convolution in NumPy in `../problems/starter.py` (reference in `../problems/_reference/solutions.py`). Build the portfolio artifact in `../../../portfolio/07_vision_classifier/`: CIFAR-10 ResNet-18 + transfer-learning baseline + Grad-CAM + FGSM.

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three paper-doable exercises (conv backprop showing $\partial L / \partial W$ is itself a conv, receptive-field recursion on a 4-layer CNN, Grad-CAM on a synthetic 2-channel feature map).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1–§3 Conv + arithmetic + backward | 4 | im2col, padding/stride; derive backward pass; verify against torch on a tiny example. |
| §4–§5 BN + ResNet | 4 | Skip connections, bottleneck; BN math; the train/eval dual behaviour. |
| §6 Transfer learning | 3 | Pretrained ImageNet backbone → frozen vs fine-tuned; the normalisation-constant footgun. |
| §7 Grad-CAM | 2 | Implement on the last block of your trained ResNet-18; eyeball 8 correct + 8 incorrect predictions. |
| §8–§9 FGSM + ViT preview | 1 | Sweep $\varepsilon$ in $\{0, 1/255, 2/255, 4/255, 8/255\}$ and plot accuracy; skim the ViT patch-embedding idea. |
| Problem set + portfolio | 5 | ResNet-18 from scratch on CIFAR-10 via `mlcourse.Trainer`; ship the failure-mode analysis. |
| Office hours / review | 1 | Cross-check against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 8, you should be able to answer "yes" to all of:

1. Can I derive conv backprop and show that both $\partial L / \partial W$ and $\partial L / \partial X$ are themselves convolutions (with $G$ and the 180°-rotated kernel respectively)?
2. Can I derive the receptive-field recursion $RF_\ell = RF_{\ell-1} + (k_\ell - 1) J_{\ell-1}$ and apply it to ResNet-18 by hand?
3. Can I explain BatchNorm's math, why train and eval modes behave differently, and what breaks at batch size 1?
4. Can I produce a Grad-CAM heatmap from scratch (pool gradients, weight feature maps, ReLU, upsample) and pick a sensible target layer?
5. Can I run an FGSM sweep and interpret the accuracy-vs-$\varepsilon$ curve (where does it drop, what does that say about decision-boundary geometry)?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **Convolutional layers ↔ translation-equivariant linear maps on a lattice.** Weight-sharing across spatial positions is exactly imposing **discrete translation symmetry** ($T_a \circ W = W \circ T_a$). Pooling is a coarse-graining / RG-block-spin step that loses some translation symmetry in exchange for a doubled lattice spacing.
- **Receptive field ↔ light cone / causal cone.** Layer depth × kernel stride determines how far an output cell can "see" — exactly the same recursion as the propagation of a signal through a discretised wave equation. Stride-2 layers are "step-2 light cones" in input-pixel time.
- **BatchNorm ↔ thermal renormalisation.** BN rescales the per-layer activations to unit variance per batch, then learns an affine to recover the right scale — same idea as RG-rescaling fields by their fluctuation scale at each block-spin step. The running-mean / running-var at eval time is the "thermalised" expectation: train statistics ≈ batch microstate, eval statistics ≈ ensemble average.
- **ResNet skip connections ↔ identity-channel propagator.** $y = x + F(x)$ at every block means the identity channel is *always* preserved through the network, so even with frozen $F$ the input can flow to the output unchanged. Same trick as adding the bare-propagator $G_0$ explicitly to a self-energy-corrected propagator $G = G_0 + G_0 \Sigma G_0 + \dots$ — preventing the network from forgetting the input is the network analogue of preserving the bare term.

Keep these bridges live; W8 (attention as an all-to-all symmetric kernel) and W10 (diffusion as a denoising flow on a U-Net) reuse the lattice / propagator pictures.
