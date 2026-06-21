#!/usr/bin/env python3
"""
iMatrix GGUF Quantization Pipeline
Produces higher-quality quantizations using importance matrix calibration.
Standard quantization rounds all weights equally — iMatrix prioritizes
important weights, giving noticeably better output at Q2/Q3/Q4.
"""

import os
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_MODEL = "google/gemma-4-12B-it"
OUTPUT_DIR = Path("./gemma4-12B-it-imatrix-GGUF")
HF_REPO = "liodon-ai/gemma-4-12B-it-imatrix-GGUF"
MODEL_DIR = Path("./model-cache")

QUANTS = [
    ("IQ2_M",  "ultra-tiny + iMatrix — better coherence than standard Q2"),
    ("IQ3_M",  "tiny + iMatrix — sharper than standard Q3"),
    ("IQ4_XS", "small + iMatrix — rivals standard Q5 at Q4 size"),
    ("Q2_K",   "tiniest standard — runs almost anywhere"),
    ("Q3_K_M", "great for 8GB VRAM"),
    ("Q4_K_M", "sweet spot (recommended)"),
    ("Q5_K_M", "high quality"),
    ("Q6_K",   "near-lossless"),
    ("Q8_0",   "basically full quality"),
]

api = HfApi()
LLAMA_DIR = Path("/tmp/llama.cpp")
QUANTIZE_BIN = LLAMA_DIR / "build/bin/llama-quantize"
IMATRIX_BIN = LLAMA_DIR / "build/bin/llama-imatrix"

# Step 1: Download base model (skip if already cached)
print("=== Checking base model ===")
MODEL_DIR.mkdir(exist_ok=True)
safetensors = list(MODEL_DIR.glob("*.safetensors"))
if safetensors:
    print(f"  Already cached ({len(safetensors)} shards), skipping download")
else:
    print("  Downloading...")
    subprocess.run(["hf", "download", BASE_MODEL, "--local-dir", str(MODEL_DIR)], check=True)

# Step 2: Build llama.cpp — must include llama-imatrix
needs_build = not QUANTIZE_BIN.exists() or not IMATRIX_BIN.exists()
if needs_build:
    print("=== Building llama.cpp (with imatrix, CPU) ===")
    if LLAMA_DIR.exists():
        shutil.rmtree(LLAMA_DIR)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_DIR)], check=True)
    subprocess.run([
        "cmake", "-B", str(LLAMA_DIR / "build"),
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ], cwd=str(LLAMA_DIR), check=True)
    subprocess.run([
        "cmake", "--build", str(LLAMA_DIR / "build"),
        "-j", str(os.cpu_count()),
        "--target", "llama-quantize", "llama-imatrix", "llama-cli",
    ], cwd=str(LLAMA_DIR), check=True)
else:
    print("=== llama.cpp already built, skipping ===")

# Step 3: Convert to F16 GGUF (skip if exists)
F16_GGUF = MODEL_DIR / "model-f16.gguf"
if F16_GGUF.exists():
    print(f"=== F16 GGUF exists ({F16_GGUF.stat().st_size/1e9:.1f} GB), skipping conversion ===")
else:
    print("=== Converting to F16 GGUF ===")
    subprocess.run([
        "python3", str(LLAMA_DIR / "convert_hf_to_gguf.py"),
        str(MODEL_DIR),
        "--outfile", str(F16_GGUF),
        "--outtype", "f16",
    ], check=True)

# Step 4: Compute iMatrix
IMATRIX_FILE = MODEL_DIR / "imatrix.dat"
CALIBRATION_FILE = LLAMA_DIR / "examples/imatrix/groups_merged.txt"

if IMATRIX_FILE.exists():
    print(f"=== iMatrix exists, skipping computation ===")
else:
    print("=== Computing importance matrix (this takes ~20-40 min) ===")
    if not CALIBRATION_FILE.exists() or CALIBRATION_FILE.stat().st_size == 0:
        print("  Generating calibration text from wikitext-2...")
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(r["text"] for r in ds if len(r["text"].strip()) > 100)[:2_000_000]
        CALIBRATION_FILE.write_text(text)
        print(f"  Calibration text: {len(text)//1000}K chars")

    subprocess.run([
        str(IMATRIX_BIN),
        "-m", str(F16_GGUF),
        "-f", str(CALIBRATION_FILE),
        "-o", str(IMATRIX_FILE),
        "--chunks", "128",
    ], check=True)
    print(f"  iMatrix saved: {IMATRIX_FILE}")

# Step 5: Quantize with iMatrix
print("=== Quantizing with iMatrix ===")
OUTPUT_DIR.mkdir(exist_ok=True)

quant_sizes = {}
for quant_name, desc in QUANTS:
    outfile = OUTPUT_DIR / f"gemma4-12B-{quant_name}.gguf"
    if outfile.exists():
        print(f"  {quant_name}: exists, skipping")
        quant_sizes[quant_name] = outfile.stat().st_size / 1e9
        continue
    print(f"  {quant_name}...")
    subprocess.run([
        str(QUANTIZE_BIN),
        "--imatrix", str(IMATRIX_FILE),
        str(F16_GGUF),
        str(outfile),
        quant_name,
    ], check=True)
    quant_sizes[quant_name] = outfile.stat().st_size / 1e9
    print(f"  -> {quant_sizes[quant_name]:.2f} GB")

# Step 6: Create model card
print("=== Creating model card ===")
readme = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
pipeline_tag: text-generation
tags:
- gguf
- imatrix
- gemma4
- local-llm
- llama.cpp
- ollama
- lm-studio
- quantized
---

# Gemma 4 12B IT — iMatrix GGUF

Higher-quality GGUF quantizations of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) using **importance matrix (iMatrix) calibration**.

## What is iMatrix?

Standard quantization rounds all weights equally. iMatrix runs a calibration pass over real text to identify which weights matter most, then prioritizes precision where it counts. The result: **noticeably better coherence and instruction-following at Q2/Q3/Q4** — same file size, better output.

The i-quants (`IQ2_M`, `IQ3_M`, `IQ4_XS`) are exclusively iMatrix-based and provide the best quality-per-GB available.

## Quick Start

### Ollama
```bash
ollama run hf.co/{HF_REPO}:Q4_K_M
```

### llama.cpp
```bash
llama-cli -hf {HF_REPO}:Q4_K_M
```

### LM Studio / Jan
Search `{HF_REPO}` and pick your quant.

## Available Quants

| Quant | Size | Notes |
|-------|------|-------|
"""

for q, d in QUANTS:
    sz = quant_sizes.get(q, 0)
    readme += f"| `{q}` | {sz:.2f} GB | {d} |\n"

readme += f"""
## VRAM Requirements

| VRAM | Recommended Quant |
|------|-------------------|
| 6 GB | `IQ2_M` |
| 8 GB | `IQ3_M` or `Q3_K_M` |
| 10 GB | `IQ4_XS` or `Q4_K_M` |
| 12 GB | `Q4_K_M` |
| 16 GB | `Q5_K_M` |
| 24 GB | `Q6_K` or `Q8_0` |

## iMatrix vs Standard — Why It Matters

At low bit widths (Q2/Q3/Q4), standard quantization loses coherence and starts producing
repetitive or broken output. iMatrix keeps the model sharp by protecting the most important
weights. If you're running at Q4 or below, prefer the iMatrix quants from this repo over
standard Q-series from other repos.

## Base Model

- **Model**: [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
- **Params**: 12B
- **Context**: 128K tokens
- **License**: Apache 2.0
- **Authors**: Google DeepMind
"""

(OUTPUT_DIR / "README.md").write_text(readme)

# Step 7: Upload
print("=== Uploading to HF ===")
create_repo(HF_REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=str(OUTPUT_DIR),
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add iMatrix GGUF quantizations",
)

print(f"\nDone! https://huggingface.co/{HF_REPO}")
