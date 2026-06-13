# Liodon AI SLM-10M

A 9.97M-parameter causal language model targeting the
[Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard) `<10M` tier.

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | 9,968,640 (9.97M) |
| Layers | 12 |
| Hidden size | 256 |
| Attention | GQA — 8 query / 2 KV heads, head_dim=32 |
| FFN | SwiGLU, intermediate=640 |
| Normalization | RMSNorm (fp32 upcast), pre-norm |
| Position encoding | RoPE θ=100,000 |
| Context length | 1,024 tokens |
| Vocabulary | 8,192 (ByteLevel BPE) |
| Weight tying | embed ↔ lm_head |
| QK-Norm | per-head, before RoPE |
| Z-loss | weight=1e-4, disabled after 31B tokens |
| Residual init | `0.02 / √(2 × 12)` for o_proj and FFN down |

## Training recipe

| Hyperparameter | Value |
|---|---|
| Total tokens | 75B |
| Batch size | 524,288 tokens (grad accum × 32 sequences) |
| Optimizer | AdamW β=(0.9, 0.95), wd=0.1 |
| Peak LR | 3e-3 |
| Min LR | 3e-4 |
| LR schedule | Warmup-stable-decay (1k / 122k / 21k steps) |
| Grad clip | 1.0 |
| Precision | bfloat16 |

### Data mix

| Source | Weight | HuggingFace dataset |
|---|---|---|
| FineWeb-Edu | 55% | `HuggingFaceFW/fineweb-edu` (sample-100BT) |
| Cosmopedia v2 | 25% | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) |
| FineWeb-HQ | 10% | `epfml/FineWeb-HQ` |
| FineMath | 10% | `HuggingFaceTB/finemath` (finemath-3plus) |

---

## Setup

```bash
git clone <this-repo>
cd slm-pretrain
pip install -r requirements.txt
```

---

## Step 1 — Train the tokenizer

Streams ~2M documents from the data sources and trains a ByteLevel BPE
tokenizer (vocab=8192). Takes ~20–40 minutes. Output: `tokenizer.json`.

```bash
python prepare_data.py --train_tokenizer
```

---

## Step 2 — Prepare the data

Downloads and tokenizes all four datasets, packing tokens into 100M-token
binary shards under `data/`. Each shard is ~200 MB (uint16). Total on disk:
~150 GB for 75B tokens.

The script is **resumable** — re-running it skips shards that already exist.

```bash
python prepare_data.py
```

To use a custom token budget (e.g. a quick 5B-token smoke run):

```bash
python prepare_data.py --total_tokens 5_000_000_000
```

---

## Step 3 — Train

```bash
python train.py
```

Checkpoints are saved to `checkpoints/step_NNNNNNN.pt` every 2,000 steps.
A `train_log.jsonl` file is written alongside.

**Resume from a checkpoint:**

```bash
python train.py --resume checkpoints/step_0010000.pt
```

**Override config fields at launch** (JSON string):

```bash
python train.py --config '{"wandb": true, "run_name": "slm-10m-run1"}'
python train.py --config '{"total_tokens": 5000000000, "compile": false}'
```

**Expected throughput on NVIDIA GB10:** ~2–3M tok/s → ~8–12 hours for 75B tokens.

---

## Step 4 — Monitor training

```bash
# Live loss tail
tail -f train_log.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if 'val_loss' in r:
        print(f\"step {r['step']:>7,}  val_loss={r['val_loss']:.4f}\")
    elif 'loss' in r:
        print(f\"step {r['step']:>7,}  loss={r['loss']:.4f}  lr={r['lr']:.2e}  {r['tok_s']/1e3:.1f}k tok/s  {r['pct']:.1f}%\")
"
```

---

## Step 5 — Export to HuggingFace format

Converts the trained checkpoint to a directory loadable via
`AutoModelForCausalLM.from_pretrained`.

```bash
python export.py --checkpoint checkpoints/final.pt --out hf_model
```

Verify the export locally:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, sys
sys.path.insert(0, 'hf_model')

model = AutoModelForCausalLM.from_pretrained('hf_model', trust_remote_code=True)
print(f'params: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## Step 6 — Run benchmarks (lm-evaluation-harness)

Install lm-eval if not already present:

```bash
pip install lm-eval
```

Run all five leaderboard benchmarks:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=hf_model,trust_remote_code=True \
  --tasks arc_easy,arc_challenge,hellaswag,piqa \
  --device cuda \
  --batch_size 64 \
  --output_path results/

# ArithMark-2 (custom benchmark)
lm_eval \
  --model hf \
  --model_args pretrained=hf_model,trust_remote_code=True \
  --tasks arithmark2 \
  --device cuda \
  --batch_size 64 \
  --output_path results/
```

---

## Step 7 — Push to HuggingFace Hub

Log in once:

```bash
huggingface-cli login
```

Push model and tokenizer:

```bash
python export.py \
  --checkpoint checkpoints/final.pt \
  --out hf_model \
  --push your-username/slm-10m
```

---

## Step 8 — Submit to the leaderboard

1. Open a PR on the [Open SLM Leaderboard Space](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard) with your benchmark results.
2. The team independently verifies the numbers and merges the PR.

Results format expected in the PR (from lm-eval output):

| Benchmark | Score |
|---|---|
| ARC-Easy (0-shot) | |
| ARC-Challenge (0-shot) | |
| HellaSwag (0-shot) | |
| PIQA (0-shot) | |
| ArithMark-2 | |
| **Average** | |

---

## File reference

| File | Purpose |
|---|---|
| `model.py` | Training model (standalone PyTorch) |
| `config.py` | All training hyperparameters |
| `tokenizer.py` | ByteLevel BPE trainer/loader |
| `prepare_data.py` | Download + tokenize → binary shards |
| `data.py` | Weighted iterable dataset over shards |
| `train.py` | Training loop |
| `configuration_slm.py` | HuggingFace `PretrainedConfig` |
| `modeling_slm.py` | HuggingFace `PreTrainedModel` wrapper |
| `export.py` | Checkpoint → HF model directory |
