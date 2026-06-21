#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen/Qwen3.5-4B for code review.
Uses bitsandbytes 4-bit quantization + LoRA on GB10 (128GB).
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3.5-4B"
DATASET_NAME = "liodon-ai/gemma4-code-review-instruct"
OUTPUT_DIR = "./qwen35-4b-reviewer-lora"
HF_REPO = "liodon-ai/qwen3.5-4B-reviewer-lora"

# LoRA params
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training params
BATCH_SIZE = 16
GRAD_ACCUM = 2
MAX_SEQ_LEN = 2048
LR = 2e-4
EPOCHS = 1

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (4-bit QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model = prepare_model_for_kbit_training(model)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, split="train")
ds_val = load_dataset(DATASET_NAME, split="validation")

SUBSET = 50000
if len(ds) > SUBSET:
    ds = ds.select(range(SUBSET))
print(f"Train: {len(ds)}, Val: {len(ds_val)}")

def tokenize_sample(sample):
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing...")
ds = ds.map(tokenize_sample, remove_columns=ds.column_names, num_proc=8)
ds_val = ds_val.map(tokenize_sample, remove_columns=ds_val.column_names, num_proc=8)

def is_valid(sample):
    return len(sample.get("input_ids", [])) > 32

ds = ds.filter(is_valid, num_proc=8)
ds_val = ds_val.filter(is_valid, num_proc=8)
print(f"After filter - Train: {len(ds)}, Val: {len(ds_val)}")

total_steps = (len(ds) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
print(f"Total training steps: ~{total_steps}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=max(10, total_steps // 20),
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=5,
    save_steps=max(100, total_steps // 5),
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=max(100, total_steps // 5),
    packing=False,
    report_to="none",
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    seed=42,
    remove_unused_columns=True,
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds_val,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Uploading to HF: {HF_REPO}")
model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)

print(f"\nDone! Uploaded to https://huggingface.co/{HF_REPO}")
