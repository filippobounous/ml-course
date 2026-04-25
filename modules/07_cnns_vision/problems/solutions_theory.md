# Week 7 — Theory-problem solutions

## 1. Conv backprop — gradient w.r.t. filter weights

For a cross-correlation output $y_{c,i,j} = \sum_{c',u,v} W_{c,c',u,v} \, x_{c', i+u, j+v} + b_c$,

$\partial L / \partial W_{c,c',u,v} = \sum_{i,j} (\partial L / \partial y_{c,i,j}) \cdot x_{c', i+u, j+v}.$

Observe the pattern: the gradient is itself a cross-correlation between the upstream gradient $\partial L/\partial y$ (playing the role of the kernel) and the input $x$ (playing the role of the feature map). So **"conv forward uses kernel $W$ against $x$; conv backward uses upstream gradient as kernel against $x$"**.

For the gradient w.r.t. the input:

$\partial L / \partial x_{c',i,j} = \sum_{c,u,v} W_{c,c',u,v} (\partial L / \partial y_{c, i-u, j-v}).$

This is a cross-correlation with the **180°-rotated** kernel (= convolution with the flipped kernel), which is why the transpose-conv / "deconv" operation is exactly this backward pass.

## 2. ResNet-18 parameter count

Stock torchvision ResNet-18 for 1000-class ImageNet (closed-form):

- Stem: 7×7×3×64 = 9,408 conv + 2·64 = 128 BN → **9,536**.
- layer1: 2 × (3·3·64·64 + 2·64 + 3·3·64·64 + 2·64) = 2 × (73,984) = **147,968**.
- layer2: downsampler block (1·1·64·128 + 2·128 + 3·3·64·128 + 2·128 + 3·3·128·128 + 2·128) = 8,448 + 73,984 + 147,712 = **230,144**, plus a plain block 3·3·128·128·2 + 2·128·2 = 295,424 → total **525,568**.
- layer3: 1·1·128·256 + 2·256 + 3·3·128·256 + 2·256 + 3·3·256·256 + 2·256 + (plain) → sums to **2,099,712**.
- layer4: analogous → **8,393,728**.
- fc: 512·1000 + 1000 = **513,000**.

Sum ≈ 11.69M, matching torchvision's `sum(p.numel() for p in resnet18().parameters()) == 11,689,512`.

Reference implementation (and the auto-check) lives in `modules/07_cnns_vision/problems/solutions.py::resnet18_param_count`.

## 3. Receptive field of ResNet-18 at each stage

Using the recursion $RF_\ell = RF_{\ell-1} + (k_\ell - 1)\, d_\ell \prod_{m<\ell} s_m$:

- Stem (7×7 stride 2): $RF = 7$, jump = 2.
- maxpool 3×3 stride 2: $RF = 7 + 2\cdot 2 = 11$, jump = 4.
- layer1 two blocks of (3×3 stride 1) × 2 each: $+4\cdot 4 = 16$ per block × 2 blocks → $RF = 11 + 32 = 43$, jump = 4.
- layer2 downsampler (3×3 stride 2 + 3×3 stride 1): $+2\cdot 4 + 2\cdot 8 = 24$, jump = 8; plain block: $+2\cdot 8 + 2\cdot 8 = 32$. $RF = 43 + 24 + 32 = 99$, jump = 8.
- layer3: analogously gives $RF = 99 + 48 + 64 = 211$, jump = 16.
- layer4: $RF = 211 + 96 + 128 = 435$, jump = 32.

A ResNet-18 has a receptive field of **~435 pixels** at the output of layer4 — larger than any 224×224 ImageNet image, i.e. every output feature has global coverage. This is why the head can be a single linear layer.

(The CIFAR stem — 3×3 stride 1 + no maxpool — shrinks these numbers drastically. That's why we use the CIFAR variant for 32×32 inputs.)
