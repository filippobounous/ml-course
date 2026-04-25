# Week 10 — Diffusion and multimodal models

## Learning objectives

1. Derive **denoising diffusion probabilistic models (DDPM)** from the variational lower bound and recognise the equivalence to **denoising score matching**.
2. Implement a **UNet-DDPM** on FashionMNIST and ship a **DDPM vs DDIM ablation** (sampling speed / quality trade-off).
3. Understand the **continuous-time view** (score SDEs, Fokker–Planck) — connects back to the Week 1 SDE primer.
4. Use pretrained **CLIP** for zero-shot retrieval and the **LLaVA** 4-bit inference path for a multimodal demo.

## Topics

- Forward / reverse diffusion processes; noise schedules (linear, cosine).
- DDPM loss = (up to constants) denoising score matching; Tweedie's formula.
- DDIM: deterministic / accelerated sampling.
- Score SDEs: VP-SDE, VE-SDE; probability-flow ODE; sampling via numerical SDE solvers.
- **CLIP**: contrastive image–text pretraining, zero-shot classification.
- Vision–language models: quick tour (LLaVA, 4-bit inference only).

## Deliverables

- Portfolio artifact: `portfolio/10_ddpm/` — DDPM vs DDIM ablation on FashionMNIST (sample quality, step counts, reproduced figure + table, mini-paper write-up). CLIP zero-shot retrieval demo as companion.

## Reading plan

See `readings.md`.
