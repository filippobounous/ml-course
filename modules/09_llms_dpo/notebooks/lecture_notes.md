# Week 9 — Language models at scale: SFT, DPO, MLX (lecture notes)

*Reading pair: Kaplan *Scaling Laws* 2020 · Hoffmann *Chinchilla* 2022 · Ouyang *InstructGPT* 2022 · Rafailov *DPO* 2023 · Hu *LoRA* 2021.*

---

## 1. Scaling laws in two paragraphs

Kaplan et al. (2020) showed that test loss on held-out tokens is a smooth power law in model parameters $N$, dataset tokens $D$, and compute $C$, with the exponents all around 0.05–0.1. The practical upshot: doubling either $N$ or $D$ buys you predictable and small amounts of loss reduction, and training beyond what the "compute-optimal" frontier predicts is wasteful.

**Chinchilla** (Hoffmann 2022) re-measured the frontier and found that compute-optimal training has $N : D$ roughly balanced (in the sense that doubling compute should roughly double both). Earlier big models (GPT-3, Gopher) were **under-trained** — they had too many parameters for the number of tokens they saw. The modern rule of thumb from Chinchilla is about **20 tokens per parameter** for dense transformers.

On a laptop we cannot verify this directly — but we can reproduce the *qualitative shape* at tiny scale by training a handful of GPT-like models at different $(N, D)$ pairs and fitting a 2-D power-law.

## 2. Supervised fine-tuning (SFT)

Start from a pretrained base LM. Fine-tune on $(prompt, response)$ pairs with a standard cross-entropy objective — but mask the loss on the prompt tokens so gradients only flow through the response.

The simplest `trl` recipe:

```python
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=TrainingArguments(output_dir="...", per_device_train_batch_size=1,
                           learning_rate=2e-5, num_train_epochs=3),
    formatting_func=lambda ex: [f"### Prompt:\n{ex['prompt']}\n### Response:\n{ex['response']}"]
)
trainer.train()
```

Two or three epochs at LR ~1e-5 to 2e-5 is usually plenty for small datasets (1–10k samples). Memorisation beyond that hurts.

## 3. Parameter-efficient fine-tuning — LoRA

LoRA (Hu 2021) augments a frozen linear layer $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ with a low-rank residual:

$$W' = W + \alpha \cdot B A, \quad A \in \mathbb{R}^{r \times d_\text{in}}, \; B \in \mathbb{R}^{d_\text{out} \times r}.$$

Only $A$ and $B$ are learned; the base weights stay frozen. Total trainable parameters per adapted layer: $r (d_\text{in} + d_\text{out})$. At $d = 4096, r = 16$ that's about **0.8% of the original count**.

- **Initialisation.** $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$. So $BA$ starts at zero and the model matches the pretrained baseline at step 0.
- **Adapted modules.** The original paper attacks just the $W_Q$ and $W_V$ projections; modern practice often adapts all four attention projections plus the MLP.
- **Merging.** At inference you can fold $BA$ back into $W$ — zero runtime overhead.

### QLoRA, and why we use MLX on Apple Silicon

QLoRA (Dettmers 2023) adds 4-bit quantisation of the base model via `bitsandbytes`. The catch: `bitsandbytes` doesn't support MPS today. So on Apple Silicon we either (a) do LoRA in fp16 on MPS via TRL, paying the memory cost of fp16 base weights, or (b) use **MLX / mlx-lm**, Apple's native framework, which supports 4-bit quantisation and LoRA fine-tuning at speeds that often beat PyTorch-MPS by 2–5×.

The command-line flow with MLX is:

```bash
python -m mlx_lm.lora --train \
  --model meta-llama/TinyLlama-1.1B-Chat-v1.0 \
  --data path/to/jsonl \
  --batch-size 1 --iters 1000 --lora-layers 16
```

## 4. RLHF, PPO, and why we skip it

RLHF with PPO (Ouyang 2022) works but is painful: three models in memory (policy, value, reference) and finicky hyperparameters. **DPO** (Rafailov 2023) shows that you can do preference optimisation **directly** on pairs, with the reward model never instantiated.

### The DPO derivation

Start from the RLHF objective over prompts $x$, completions $y$:

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta D_\text{KL}(\pi(y | x) \| \pi_\text{ref}(y | x)).$$

The closed-form optimal policy is

$$\pi^\star(y | x) = \frac{1}{Z(x)} \pi_\text{ref}(y | x) \exp\!\left(\tfrac{1}{\beta} r(x, y)\right).$$

Rearranging:

$$r(x, y) = \beta \log \frac{\pi^\star(y | x)}{\pi_\text{ref}(y | x)} + \beta \log Z(x).$$

With a Bradley–Terry preference model $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$, the $\log Z(x)$ cancels (it's a function of $x$ only), leaving the **DPO loss**:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[ \log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} - \beta \log\frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)} \right) \right].$$

The optimisation target is $\pi_\theta$ initialised at $\pi_\text{ref}$; $\pi_\text{ref}$ is frozen. Only a ratio of log-probs plus a sigmoid + log: almost trivial to implement.

### Successors (skim)

- **ORPO** (Hong 2024) folds SFT and preference optimisation into a single loss — simpler and often competitive.
- **KTO** (Ethayarajh 2024) works with unpaired preferences (thumbs-up / thumbs-down).
- **IPO, SimPO, CPO** — lots of ablation space; DPO is still the sensible default.

## 5. Eval harnesses

Aggregate benchmarks (MMLU, GSM8K, MT-Bench) exist but are weak proxies for "is this model actually good at my task?". For a portfolio artifact, build a **task-specific rubric**:

1. Curate 20–50 prompts that target the behaviour you changed (e.g. helpfulness, refusal behaviour, factual accuracy in your domain).
2. Score pairs with either (a) an external LLM judge (Claude / GPT-4) or (b) a deterministic rule + self-consistency.
3. Report **win-rate** of your tuned model over the SFT baseline.

Treat "LLM-as-judge" numbers as ordinal, not cardinal: they're useful for ranking, not for headline accuracy claims.

## 6. Model cards and dataset cards

Mitchell et al. (2019) and Gebru et al. (2018) describe the minimum docs every model you publish should carry: intended use, not-intended use, performance on subpopulations, known failure modes, training-data provenance. `huggingface_hub` has `ModelCard` and `DatasetCard` helpers — actually use them.

## What to do with these notes

Work the problem set in `../problems/README.md`. The portfolio artifact this
week is in `../../../portfolio/09_dpo_tinyllama/`: SFT + DPO on
TinyLlama-1.1B-Chat via TRL or MLX + eval harness + model card + Gradio demo.
