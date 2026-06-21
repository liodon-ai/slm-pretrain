---
license: apache-2.0
base_model: google/gemma-4-12B-it
pipeline_tag: text-generation
tags:
- gguf
- gemma4
- local-llm
- llama.cpp
- ollama
- conversational
- reasoning
- coding
- thinking
- any-to-any
language:
- en
---

# Gemma 4 12B IT — GGUF Quantizations

GGUF quantizations of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) by Google DeepMind.

## Model Overview

Gemma 4 12B IT is a 12-billion parameter instruction-tuned model from Google DeepMind, built on the Gemma 4 architecture. It features:

- **Reasoning** — Configurable thinking modes for step-by-step problem solving
- **Coding** — Strong code generation, completion, and correction capabilities
- **Long Context** — 256K token context window
- **Multilingual** — Support for 140+ languages
- **Function Calling** — Native tool use support for agentic workflows
- **License** — Apache 2.0 (free to use, modify, and redistribute)

These GGUF quantizations enable running the model locally on consumer hardware using `llama.cpp`, Ollama, LM Studio, and other compatible tools.

## Quick Start

### Ollama
```bash
ollama run hf.co/liodon-ai/gemma-4-12B-it-GGUF:Q4_K_M
```

### llama.cpp
```bash
# Install llama.cpp
brew install llama.cpp  # macOS
# or download from https://github.com/ggerganov/llama.cpp/releases

# Start server with web UI
llama-server -hf liodon-ai/gemma-4-12B-it-GGUF:Q4_K_M

# Or run directly in terminal
llama-cli -hf liodon-ai/gemma-4-12B-it-GGUF:Q4_K_M
```

### LM Studio
1. Open LM Studio
2. Search for `liodon-ai/gemma-4-12B-it-GGUF`
3. Download your preferred quantization
4. Start chatting

### Jan
1. Open Jan
2. Navigate to Hub
3. Search `liodon-ai/gemma-4-12B-it-GGUF`
4. Download and run

## Available Quantizations

| Quant | File Size | Quality | Best For |
|-------|-----------|---------|----------|
| `Q2_K` | ~4.8 GB | Lowest — usable | Ultra-low VRAM (6GB), testing |
| `Q3_K_M` | ~6.1 GB | Good — much better than Q2 | 8GB VRAM GPUs |
| `Q4_K_M` | ~7.4 GB | **Sweet spot** (recommended) | 8-12GB VRAM, best balance |
| `Q5_K_M` | ~8.6 GB | High quality | 12GB VRAM, near-lossless |
| `Q6_K` | ~9.8 GB | Near-lossless | 16GB VRAM, high fidelity |
| `Q8_0` | ~12.7 GB | Basically full quality | 24GB VRAM, maximum quality |

## Hardware Requirements

Estimated VRAM requirements for full model loading (no context):

| VRAM | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|------|------|--------|--------|--------|------|------|
| **6 GB** | ✓ | — | — | — | — | — |
| **8 GB** | ✓ | ✓ | tight | — | — | — |
| **12 GB** | ✓ | ✓ | ✓ | tight | — | — |
| **16 GB** | ✓ | ✓ | ✓ | ✓ | tight | — |
| **24 GB** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> **Tip:** Use `--cache-type-k q4_0 --cache-type-v q4_0` in llama.cpp to roughly double your available context length.

## Context Length Cheat Sheet

Rough estimates for max context length at each quantization (assumes `q8_0` KV cache + ~1.5 GB overhead):

| VRAM | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|------|------|--------|--------|--------|------|------|
| **8 GB** | ~16K | ~10K | ~2-4K | — | — | — |
| **12 GB** | ~48K | ~38K | ~30K | ~20K | ~12K | — |
| **16 GB** | ~80K | ~72K | ~64K | ~52K | ~44K | ~22K |
| **24 GB** | ~200K | ~160K | ~128K | ~110K | ~90K | ~60K |
| **32 GB** | 256K (max) | 256K | 256K | 256K | ~230K | ~190K |

## Recommended Sampling Parameters

For best results with Gemma 4:

| Mode | Temperature | Top P | Top K | Use Case |
|------|-------------|-------|-------|----------|
| **General** | 1.0 | 0.95 | 64 | Chat, creative tasks |
| **Coding** | 0.6 | 0.95 | 20 | Code generation |
| **Deterministic** | 0.0 | 1.0 | 1 | Reproducible outputs |
| **Reasoning** | 1.0 | 0.95 | 64 | Math, logic puzzles |

## Thinking Mode

Gemma 4 supports configurable thinking modes. Enable thinking for complex tasks:

- **Enable thinking**: Add `<|think|>` token at the start of the system prompt
- **Disable thinking**: Remove the `<|think|>` token
- **Default**: Most libraries handle this automatically via the chat template

## Model Architecture

| Property | Value |
|----------|-------|
| Architecture | Gemma4Unified |
| Parameters | 12B |
| Layers | 48 |
| Hidden Size | 3,840 |
| Attention Heads | 16 (Q) / 8 (KV) |
| Context Length | 256K tokens |
| Vocabulary | 262,144 |
| Sliding Window | 1,024 tokens |

## Base Model

- **Model**: [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)
- **Organization**: Google DeepMind
- **License**: [Apache 2.0](https://ai.google.dev/gemma/apache_2)
- **Blog**: [Gemma 4 Launch](https://blog.google/technology/developers/gemma-4/)

## Quantization Method

Quantized using `llama.cpp`'s `llama-quantize` tool with the following methods:
- **Q2_K** — 2-bit K-quants (super-block size 16)
- **Q3_K_M** — 3-bit K-quants (medium quality)
- **Q4_K_M** — 4-bit K-quants (medium quality, recommended)
- **Q5_K_M** — 5-bit K-quants (medium quality)
- **Q6_K** — 6-bit K-quants (all tensors quantized to 6-bit)
- **Q8_0** — 8-bit block-wise quantization (near-lossless)

## Citation

```bibtex
@misc{gemma4_12b_it_gguf,
  title = {Gemma 4 12B IT GGUF Quantizations},
  author = {{liodon-ai}},
  year = {2026},
  url = {https://huggingface.co/liodon-ai/gemma-4-12B-it-GGUF},
  note = {Quantizations of google/gemma-4-12B-it by Google DeepMind}
}
```
