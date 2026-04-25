"""SFT + DPO fine-tuning of TinyLlama-1.1B-Chat via HuggingFace TRL.

This script is a thin wrapper around `trl.SFTTrainer` and `trl.DPOTrainer`
configured for LoRA + fp16 on MPS. For Apple Silicon, prefer the `mlx-lm`
path described in this folder's README.md — it's 2-5× faster than TRL on MPS.

Requires `pip install -e '.[dl,llm,ops]'`. Downloading TinyLlama
(~1.1B params → ~2.2 GB in fp16) and a preference dataset requires network
access and a HuggingFace token (`huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft-dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--dpo-dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--sft-samples", type=int, default=2000)
    parser.add_argument("--dpo-samples", type=int, default=2000)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="1-epoch 50-sample smoke run")
    args = parser.parse_args()

    if args.quick:
        args.sft_samples = 50
        args.dpo_samples = 50
        args.sft_epochs = 1
        args.dpo_epochs = 1

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    from mlcourse.utils import detect_device

    device = detect_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    base.to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -- SFT ------------------------------------------------------------------
    sft_ds = load_dataset(args.sft_dataset, split=f"train[:{args.sft_samples}]")

    def format_sft(ex):
        prompt = ex.get("instruction") or ex.get("prompt", "")
        response = ex.get("output") or ex.get("response", "")
        return {"text": f"### Instruction:\n{prompt}\n### Response:\n{response}"}

    sft_ds = sft_ds.map(format_sft, remove_columns=sft_ds.column_names)
    sft_args = SFTConfig(
        output_dir=str(Path(args.output_dir) / "sft"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.sft_epochs,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="no",
        fp16=(device != "cpu"),
        report_to="none",
    )
    sft = SFTTrainer(
        model=base,
        args=sft_args,
        train_dataset=sft_ds,
        peft_config=lora_cfg,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    print("Starting SFT...")
    sft.train()
    sft.save_model(str(Path(args.output_dir) / "sft"))

    # -- DPO ------------------------------------------------------------------
    dpo_ds = load_dataset(args.dpo_dataset, split=f"train_prefs[:{args.dpo_samples}]")
    dpo_args = DPOConfig(
        output_dir=str(Path(args.output_dir) / "dpo"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.dpo_epochs,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=20,
        save_strategy="no",
        fp16=(device != "cpu"),
        report_to="none",
    )
    dpo = DPOTrainer(
        model=base,
        ref_model=None,  # TRL uses the frozen base weights implicitly via PEFT
        args=dpo_args,
        train_dataset=dpo_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
    )
    print("Starting DPO...")
    dpo.train()
    dpo.save_model(str(Path(args.output_dir) / "dpo"))

    print(f"\nDone. SFT + DPO LoRA adapters saved under {args.output_dir}.")


if __name__ == "__main__":
    sys.exit(main())
