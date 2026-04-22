# 10 — DDPM vs DDIM ablation

> Populated in Week 10. See `modules/10_diffusion_multimodal/`.

## Problem
Train a small UNet-DDPM on FashionMNIST; produce an honest DDPM vs DDIM
ablation on sample quality vs sampling steps.

## Method
- UNet-small (~5M params) trained with the ε-prediction objective.
- Sampling via DDPM (1000 steps) and DDIM (10 / 20 / 50 / 100 steps, η=0).
- FID / IS reported on 5k generated samples.

## Results
*Sample grids, FID table, mini-paper PDF.*

## Reproduce
```bash
make -C portfolio/10_ddpm reproduce
```
