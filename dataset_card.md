---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- code-review
- code
- reasoning
- chain-of-thought
- think-tags
- instruction-tuning
- software-engineering
- python
- typescript
- javascript
- java
- go
- rust
- github
- pull-request
size_categories:
- 100K<n<1M
viewer: true
---

# gemma4-code-review-instruct

**197K code review examples — 58K with chain-of-thought `<think>` reasoning traces.**

Built to train models that don't just flag issues, but *explain their reasoning* before delivering a review. Drop-in ready for SFT with any chat model.

## Why This Dataset

Most code review datasets give you `diff → comment`. This one gives you `diff → think → comment` for 30% of examples — reasoning traces that show *how* to analyze a diff before writing the review. That signal teaches models to catch non-obvious bugs, not just style issues.

## Dataset Statistics

| Split | Examples |
|-------|----------|
| Train | 187,226 |
| Validation | 9,855 |
| **Total** | **197,081** |

| Subset | Examples | Format |
|--------|----------|--------|
| With `<think>` reasoning traces | ~58,600 | diff → `<think>...</think>` → review |
| Standard reviews | ~138,500 | diff → review |

## Data Format

All examples use the standard `messages` format, compatible with any chat template:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert code review assistant. First think through the code changes step by step, then provide a clear, actionable review. Use <think> tags for your reasoning before giving the final review."
    },
    {
      "role": "user",
      "content": "Language: Python\n\nDiff:\n```diff\n@@ -14,6 +14,8 @@\n-    return db.query(User).filter(User.id == user_id).first()\n+    user = db.query(User).filter(User.id == user_id).first()\n+    cache[user_id] = user\n+    return user\n```"
    },
    {
      "role": "assistant",
      "content": "<think>\nThe diff adds a cache write after a DB query. Need to check: 1) Is `cache` defined in scope? 2) No TTL — stale data risk. 3) No thread safety — concurrent writes could corrupt the dict. 4) Cache is written even if user is None.\n</think>\n\nThe caching logic has several issues:\n- **Stale data**: no TTL means updates to the user won't be reflected until restart\n- **Thread safety**: `cache[user_id] = user` is not atomic; use `threading.Lock` or switch to a thread-safe cache\n- **None caching**: if the user doesn't exist, `None` gets cached and future lookups will return `None` without hitting the DB"
    }
  ],
  "source": "reasoning-sft-github-codereview"
}
```

## Sources

| Source | Rows (raw) | After filtering | Description |
|--------|-----------|-----------------|-------------|
| [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview) | 334K | ~138K | Real GitHub PR comments with diff context, language tags, quality scores |
| [AmanPriyanshu/reasoning-sft-github-codereview](https://huggingface.co/datasets/AmanPriyanshu/reasoning-sft-github-codereview) | 76.7K | ~58K | Code review with explicit reasoning traces, categorized by review type |

## Processing Pipeline

1. **Normalization** — unified `messages` format across all sources
2. **Quality filtering** — removed low-quality comments (score < 0.3) and flagged-negative examples
3. **Reasoning augmentation** — `<think>` tags injected for reasoning-source examples; preserved where present
4. **Deduplication** — MD5-based dedup on user message content
5. **Length filtering** — examples > 8K estimated tokens removed
6. **Split** — 95/5 train/validation

## Languages Covered

Python · TypeScript · JavaScript · Java · C++ · Go · Rust · and more from real GitHub PR data

## Use Cases

- SFT for code review assistants (works with Qwen, Gemma, Llama, Mistral)
- Training models that reason before responding
- Automated PR review pipelines
- Code quality and security analysis tools

## Related Model

Fine-tuned on this dataset: [`liodon-ai/qwen3.5-4B-reviewer-lora`](https://huggingface.co/liodon-ai/qwen3.5-4B-reviewer-lora)

## License

Apache 2.0

## Citation

```bibtex
@misc{gemma4_code_review_instruct_2026,
  title  = {gemma4-code-review-instruct: Code Review Instruction-Tuning Dataset with Reasoning Traces},
  author = {{liodon-ai}},
  year   = {2026},
  url    = {https://huggingface.co/datasets/liodon-ai/gemma4-code-review-instruct}
}
```
