# Problem set — Week 10

## Theory

1. **DDPM ELBO.** Derive the variational lower bound for DDPM and reduce the per-step loss to the simple $\|\epsilon - \epsilon_\theta\|^2$ form.
2. **Score ≡ denoising.** Show that denoising score matching is equivalent (up to a data-independent constant) to the DDPM training objective.
3. **DDIM determinism.** Derive the DDIM update and show that $\eta = 0$ gives a deterministic map corresponding to the probability-flow ODE.
4. **CLIP contrastive.** Write the InfoNCE loss for a batch of $N$ paired image–text examples; identify its two softmax normalisations.

## Implementation (portfolio)

5. Implement a **UNet-small (~5M params)** DDPM on FashionMNIST. Train to generate plausible samples in ≤ 3 hours on MPS.
6. **DDIM vs DDPM ablation**: sweep over step counts $\{10, 20, 50, 100, 1000\}$; report FID and visual samples. Produce the ablation table for the mini-paper.
7. **CLIP zero-shot retrieval**: use `open_clip` ViT-B/32 to build a small image search over a personal / Pexels image folder.

## Applied

8. **LLaVA demo** (inference-only): use MLX / llama.cpp to run LLaVA-1.5-7B at 4-bit and caption five images. Short write-up of qualitative findings.

## Grading

Tests in `tests/week_10/` check: forward-noise schedule matches the closed form $\bar{\alpha}_t$; DDIM sampler with $\eta=0$ is deterministic; `open_clip` image embeddings have unit norm after normalisation.
