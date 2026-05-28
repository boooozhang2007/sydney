"""Export an Unsloth fine-tuning notebook for the accepted dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

TARGET_MODELS: Dict[str, str] = {
    "qwen36_27b": "unsloth/Qwen3.6-27B",
    "gemma4_31b": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
}

TARGET_LABELS: Dict[str, str] = {
    "qwen36_27b": "Qwen3.6-27B",
    "gemma4_31b": "Gemma 4 31B",
}


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def export_unsloth_notebook(
    output_path: str | Path,
    *,
    target_model_key: str = "qwen36_27b",
    train_mode: str = "qlora",
    dataset_filename: str = "train_chatml.jsonl",
) -> Path:
    """生成一个可直接打开的 ipynb。

    train_mode:
      - qlora: 4-bit + LoRA，默认推荐
      - lora: 16-bit/bf16 LoRA，需要更大显存
      - fft: full fine-tuning，需要非常大的显存/存储
    """

    if target_model_key not in TARGET_MODELS:
        raise ValueError(f"未知目标模型：{target_model_key}")
    if train_mode not in {"qlora", "lora", "fft"}:
        raise ValueError("train_mode 必须是 qlora/lora/fft")

    model_name = TARGET_MODELS[target_model_key]
    target_label = TARGET_LABELS[target_model_key]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        _md(
            f"""# Sydney 风格微调 Notebook

目标模型：`{model_name}`（{target_label}）
训练模式：`{train_mode}`
数据文件：`{dataset_filename}`

> 这个 notebook 由本地 Sydney 数据工厂自动生成。请在有足够 GPU 显存的环境运行；本地工作台不加载 27B/31B 模型。
"""
        ),
        _code(
            """# 环境安装（Colab / 云 GPU / Linux 环境）。如你已有环境，可跳过或按需调整。
!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
!pip install -U datasets trl accelerate bitsandbytes transformers peft
"""
        ),
        _code(
            f"""import os
import torch

TARGET_MODEL_KEY = {target_model_key!r}
MODEL_NAME = {model_name!r}
TRAIN_MODE = {train_mode!r}  # qlora / lora / fft
DATASET_PATH = {dataset_filename!r}
OUTPUT_DIR = f"./sydney-{{TARGET_MODEL_KEY}}-{{TRAIN_MODE}}"

MAX_SEQ_LENGTH = 4096
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4 if TRAIN_MODE in ("qlora", "lora") else 1e-5
LORA_RANK = 16
LORA_ALPHA = 32

# 可选：私有/门控模型需要 HF token。
# os.environ["HF_TOKEN"] = "hf_xxx"
"""
        ),
        _code(
            """from datasets import load_dataset

raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(raw_dataset)
print(raw_dataset[0])
"""
        ),
        _code(
            """from unsloth import FastModel
from unsloth import is_bfloat16_supported

load_in_4bit = TRAIN_MODE == "qlora"
load_in_8bit = False
full_finetuning = TRAIN_MODE == "fft"

model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    full_finetuning = full_finetuning,
    token = os.environ.get("HF_TOKEN") or None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
"""
        ),
        _code(
            """# 把 ChatML messages 渲染为模型 chat template 文本。
def to_messages(example):
    messages = example.get("messages")
    if messages is None and "conversations" in example:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": role_map.get(x["from"], x["from"]), "content": x["value"]}
            for x in example["conversations"]
        ]
    return messages


def render_chat(example):
    messages = to_messages(example)
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
            # Qwen 系模型如需关闭 thinking，可在支持该参数时启用：
            # enable_thinking = False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


dataset = raw_dataset.map(render_chat, remove_columns=raw_dataset.column_names)
print(dataset[0]["text"][:1200])
"""
        ),
        _code(
            """# LoRA / QLoRA 只训练 adapter；FFT 跳过 adapter 注入。
if TRAIN_MODE in ("qlora", "lora"):
    try:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers = False,
            finetune_language_layers = True,
            finetune_attention_modules = True,
            finetune_mlp_modules = True,
            r = LORA_RANK,
            lora_alpha = LORA_ALPHA,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
    except TypeError:
        # 兼容纯文本 LLM API。
        model = FastModel.get_peft_model(
            model,
            r = LORA_RANK,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = LORA_ALPHA,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = MAX_SEQ_LENGTH,
        )
else:
    print("FFT 模式：不注入 LoRA adapter，将训练完整权重。请确认显存和存储足够。")
"""
        ),
        _code(
            """from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    args = SFTConfig(
        max_seq_length = MAX_SEQ_LENGTH,
        per_device_train_batch_size = PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        learning_rate = LEARNING_RATE,
        warmup_ratio = 0.03,
        logging_steps = 1,
        optim = "adamw_8bit" if TRAIN_MODE in ("qlora", "lora") else "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        report_to = "none",
    ),
)
trainer.train()
"""
        ),
        _code(
            """# 保存训练产物。
if TRAIN_MODE in ("qlora", "lora"):
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")
else:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Full checkpoint saved to {OUTPUT_DIR}")
"""
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
