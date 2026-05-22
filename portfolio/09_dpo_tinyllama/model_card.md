# Model card — DPO-tuned TinyLlama-1.1B-Chat

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Base.** [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
- **Adaptation.** SFT (LoRA $r=16$, 4 attention projections + MLP) on
  an instruction dataset, then DPO ($\beta = 0.1$) on $\le 5$k
  UltraFeedback preference pairs.
- **Framework.** HuggingFace TRL on MPS (PyTorch path) **or**
  `mlx-lm` (Apple Silicon native, 4-bit base) — both supported.
- **Hyperparameters.** SFT: 2 epochs, LR 2e-5. DPO: 1 epoch, LR 5e-7,
  $\beta = 0.1$.

## Intended use

- **Primary.** Demonstrate modern LLM alignment (DPO, not PPO) on a
  laptop-scale base model.
- **Out-of-scope.** Production deployment. TinyLlama at 1.1B is too
  small for reliable instruction following; the DPO update sharpens
  helpfulness/refusal at this scale but does not unlock genuinely
  new capabilities.

## Metrics

- **DPO loss** on a held-out preference split.
- **Win-rate** of DPO model vs SFT baseline on 20–30 curated prompts,
  judged by an external LLM (or self-consistency if offline).
- **Treat all "X% win-rate" numbers as ordinal, not cardinal** (see
  W9 lecture notes).

## Training / evaluation data

- **SFT.** Alpaca-cleaned *or* UltraChat-subset (configurable). Both
  English-only; both have well-known limitations (synthetic outputs,
  occasional refusal-bait).
- **DPO.** UltraFeedback (≤ 5k preference pairs).
- **Eval prompts.** Hand-curated; not derived from the train split.

## Quantitative analyses

| Metric | Target | Verified |
|---|---|---|
| Win-rate vs SFT baseline | 55–60% | ⏳ (aspirational; needs hardware run) |
| Training runtime (MPS) | ~3 h SFT + ~1 h DPO | ⏳ |

## Ethical considerations

- The DPO objective optimises the preference signal under the
  Bradley–Terry model; it does **not** make the model truthful, safe,
  or aligned with broader human values. A 60% win-rate over SFT on a
  helpfulness rubric is consistent with substantial residual failure
  modes (jailbreaks, hallucinations).
- LLM-as-judge introduces the judge's own biases into the eval. If
  the judge is GPT-4-class, the eval has GPT-4's blind spots
  baked in.

## Caveats

- The base model is 1.1B; nothing said in the W9 lecture notes
  about RLHF at frontier scale (GPT-4-class, Anthropic Constitutional
  AI) applies *directly* here.
- DPO is sensitive to $\beta$. The 55–60% target assumes
  $\beta = 0.1$; expect different numbers at $\beta \in \{0.01, 0.5\}$.
- Single-seed run. Re-run with at least 3 seeds before quoting a
  win-rate number publicly.
