# Week 7 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Conv backprop on a tiny input

Forward (cross-correlation, no padding, stride 1). Input $X$ is $3 \times 3$, kernel $W$ is $2 \times 2$:

$$
X = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}, \quad W = \begin{pmatrix} w_{00} & w_{01} \\ w_{10} & w_{11} \end{pmatrix}.
$$

Output $Y$ is $2 \times 2$:

$$
Y = \begin{pmatrix} y_{00} & y_{01} \\ y_{10} & y_{11} \end{pmatrix}, \quad y_{ij} = \sum_{u, v} W_{uv}\, X_{i+u, j+v}.
$$

Explicitly: $y_{00} = w_{00} a + w_{01} b + w_{10} d + w_{11} e$.

### Backward: $\partial L / \partial W$

$$
\frac{\partial L}{\partial w_{uv}} = \sum_{i, j} \frac{\partial L}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial w_{uv}} = \sum_{i, j} \frac{\partial L}{\partial y_{ij}} \cdot X_{i+u, j+v}.
$$

Substitute upstream gradient $G_{ij} := \partial L / \partial y_{ij}$:

$$
\frac{\partial L}{\partial W} = G \star X \quad \text{(cross-correlation of $G$ against $X$)}.
$$

For $w_{00}$ specifically: $G_{00} a + G_{01} b + G_{10} d + G_{11} e$. The same algebraic pattern as the forward pass, but with $G$ playing the role of "kernel" and $X$ the "input".

### Backward: $\partial L / \partial X$

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{u, v} W_{uv} \cdot G_{i-u, j-v}.
$$

That's a cross-correlation with the **180°-rotated** kernel $\tilde W_{uv} = W_{1-u, 1-v}$, i.e. a *full convolution* (in the strict mathematical sense). This is exactly what `nn.ConvTranspose2d` does, and the reason the "transpose conv" is sometimes (sloppily) called "deconvolution".

### Take-home

Both the forward op and its two backward ops are convolutions. Conv is closed under autodiff — which is why GPU conv kernels are so heavily optimised.

---

## Example 2 — Receptive field of a small CNN

Layer stack: input → conv(3×3, stride 1) → conv(3×3, stride 2) → conv(3×3, stride 1) → conv(5×5, stride 1).

Recursion: $RF_\ell = RF_{\ell-1} + (k_\ell - 1) \cdot J_{\ell-1}$ and $J_\ell = J_{\ell-1} \cdot s_\ell$, with $RF_0 = 1$, $J_0 = 1$.

| Layer | $k$ | $s$ | $RF$ before | $J$ before | $RF$ after | $J$ after |
|---|---|---|---|---|---|---|
| 1 (conv 3×3, s=1) | 3 | 1 | 1 | 1 | $1 + (3-1)\cdot 1 = 3$ | $1 \cdot 1 = 1$ |
| 2 (conv 3×3, s=2) | 3 | 2 | 3 | 1 | $3 + 2 \cdot 1 = 5$ | $1 \cdot 2 = 2$ |
| 3 (conv 3×3, s=1) | 3 | 1 | 5 | 2 | $5 + 2 \cdot 2 = 9$ | $2 \cdot 1 = 2$ |
| 4 (conv 5×5, s=1) | 5 | 1 | 9 | 2 | $9 + 4 \cdot 2 = 17$ | $2 \cdot 1 = 2$ |

Final RF = $17 \times 17$ input pixels per output cell. Note how the
*stride-2 layer doubles future kernel sizes* in input-pixel terms —
that's why downsampling early grows the RF cheaply, and why ResNets
use stride-2 in `layer2`, `layer3`, `layer4`.

### Compare to the W7 problem set

The same recursion takes ResNet-18 from $RF = 7$ (stem) to $RF = 435$
by the end of `layer4` — covering the entire 32×32 CIFAR input
*twice over*, which is why the classifier sees global context.

---

## Example 3 — Grad-CAM on a synthetic feature map

Setup: a single conv layer produces a $C \times H \times W$ feature
map. Set $C = 2$, $H = W = 4$. Pretend the class logit is
$y^c = \text{globalavg}(A^0) - \text{globalavg}(A^1)$ — i.e. the
model believes channel 0 supports class $c$ and channel 1 opposes it.

### Step 1. Pooled gradients ⟹ $\alpha_k^c$

$$
\alpha_k^c = \frac{1}{HW} \sum_{i, j} \frac{\partial y^c}{\partial A^k_{ij}}.
$$

By construction $\partial y^c / \partial A^0_{ij} = 1/(HW) = 1/16$ and
$\partial y^c / \partial A^1_{ij} = -1/16$. So
$\alpha_0^c = 1/16$, $\alpha_1^c = -1/16$.

### Step 2. Weighted sum + ReLU

$$
L^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right) = \text{ReLU}\!\left(\tfrac{1}{16}(A^0 - A^1)\right).
$$

Where channel 0 is *active and* channel 1 is *quiet*, $L^c$ is large
and positive. Where the reverse holds, $L^c$ is zeroed by ReLU.

### Step 3. Upsample to input resolution

Bilinear upsample $4 \times 4 \to H_\text{img} \times W_\text{img}$,
overlay on the input as a heatmap.

### Sanity check

If you swap which channel supports class $c$ (so $y^c = -\text{avg}(A^0) + \text{avg}(A^1)$), the signs flip, the ReLU kills the previously-bright region, and the new bright region is exactly the area where $A^1$ exceeds $A^0$. The localisation map is **class-discriminative** — that's what makes Grad-CAM useful as a diagnostic.

### Failure mode

If $\partial y^c / \partial A^k = 0$ for every $k$ (e.g., logit is constant w.r.t. that layer's features), every $\alpha_k^c = 0$ and Grad-CAM produces a blank map. Common cause: you picked a target layer that's too shallow to encode the class concept. Pick the last conv block.

---

## What to do with these examples

For Example 1, work the same algebra with stride 2 and notice how the
backward ops grow correspondingly. For Example 2, add a 7×7 stem to
the front (as in real ResNets) and see how dramatically the RF jumps
in a single layer. For Example 3, run a real Grad-CAM on a
correctly-classified CIFAR image and an incorrectly-classified one,
and compare — that's the actual W7 portfolio artifact.
