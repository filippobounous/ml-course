# Model Card — TinyLlama-1.1B-Chat-DPO (my-username)

**A DPO-tuned variant of TinyLlama-1.1B-Chat-v1.0 produced as coursework for
[ml-course](../../README.md).**

## Model details

- **Base model.** TinyLlama-1.1B-Chat-v1.0 (Zhang et al. 2024).
- **Fine-tuning method.** Supervised fine-tuning + Direct Preference
  Optimization (DPO, Rafailov et al. 2023) via LoRA (r=16, α=32, all
  attention projections).
- **Compute.** <N> GPU-hours on <platform> (Apple M-series MPS / CUDA).
- **Precision.** fp16 base weights, fp32 LoRA adapters.

## Intended use

- **Primary.** <one-sentence description of the behaviour this adapter
  targets — e.g. "shorter, more direct answers on general-knowledge
  questions">.
- **Out of scope.** Safety-critical domains, medical / legal advice,
  high-stakes decision making. This is a coursework model, not a production
  model.

## Training data

- **SFT.** yahma/alpaca-cleaned (first <N> samples).
- **DPO.** HuggingFaceH4/ultrafeedback_binarized train_prefs (first <N>
  samples).
- Licences: see upstream dataset cards.

## Evaluation

- **Evaluation set.** <N> curated prompts covering {list behaviour slices}.
- **Judge.** <external LLM / deterministic rubric>.
- **Result.** Win-rate over SFT baseline: <XX>% (<XX>% 95% CI).

## Known limitations

- Hallucinates on factual queries outside the SFT / DPO distribution.
- Small base model (1.1B params) — cannot reason at Llama-3-70B levels.
- LoRA adapter only — not merged into the base weights.

## Bias / safety

- Reflects biases present in the base model and training datasets; no safety
  filtering was applied beyond what UltraFeedback provides.
- Do not deploy without your own safety evaluation for the intended use case.

## Reproducibility

- Code: `portfolio/09_dpo_tinyllama/` in this repository.
- Seed: `mlcourse.utils.seed_everything(0, deterministic_torch=True)`.
- Exact hyperparameters: see `output/dpo/config.yaml` produced by the trainer.

## Citation

```bibtex
@misc{ml-course-dpo-tinyllama,
  author = {<Your Name>},
  title  = {TinyLlama-1.1B-Chat-DPO (ml-course)},
  year   = {2026},
  howpublished = {HuggingFace Hub, \url{https://huggingface.co/<user>/<repo>}},
}
```
