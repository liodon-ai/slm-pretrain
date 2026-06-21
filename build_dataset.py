#!/usr/bin/env python3
"""
Data pipeline for gemma-4-12B-reviewer fine-tuning.
Loads, normalizes, deduplicates, and merges code review datasets from HF.
"""

import hashlib
import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = (
    "You are an expert code review assistant. Analyze the provided code diff and "
    "provide constructive, specific, and actionable review comments. Focus on bugs, "
    "performance issues, security concerns, code style, and best practices. "
    "Be concise but thorough."
)

SYSTEM_PROMPT_REASONING = (
    "You are an expert code review assistant. First think through the code changes "
    "step by step, then provide a clear, actionable review. Use <think> tags for "
    "your reasoning before giving the final review."
)


def normalize_github_codereview(ds):
    """ronantakizawa/github-codereview: diff + reviewer_comment -> review"""
    records = []
    for row in ds:
        diff = row.get("diff_context", "")
        comment = row.get("reviewer_comment", "")
        before = row.get("before_code", "")
        after = row.get("after_code", "")
        lang = row.get("language", "")
        quality = row.get("quality_score", 0)
        is_negative = row.get("is_negative", False)

        if not diff or not comment:
            continue
        if is_negative:
            continue
        if quality < 0.3:
            continue

        user_content = f"Language: {lang}\n\nDiff:\n```diff\n{diff}\n```"
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": comment},
            ],
            "source": "github-codereview",
        })
    return records


def normalize_reasoning_sft(ds):
    """AmanPriyanshu/reasoning-sft-github-codereview: input/response with reasoning"""
    records = []
    for row in ds:
        input_msgs = row.get("input", [])
        response = row.get("response", "")
        category = row.get("category", "")

        if not input_msgs or not response:
            continue

        # Reconstruct: extract the code/diff from user message
        user_msg = ""
        for msg in input_msgs:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        if not user_msg:
            continue

        # Add reasoning tags if not present
        if "<think>" not in response:
            response = f"<think>\nAnalyzing the code changes...\n</think>\n\n{response}"

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "source": "reasoning-sft-github-codereview",
        })
    return records


def normalize_dahoas(ds):
    """Dahoas/code-review-instruct-critique-revision: instruct format"""
    records = []
    for row in ds:
        # Try to find instruction/response columns
        instruction = row.get("instruction", "") or row.get("prompt", "") or row.get("input", "")
        response = row.get("response", "") or row.get("output", "") or row.get("answer", "")

        if not instruction or not response:
            continue

        # Check if it looks like a code review task
        if not any(kw in instruction.lower() for kw in ["review", "code", "diff", "pr", "pull request"]):
            continue

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "source": "dahoas-code-review",
        })
    return records


def normalize_agent_traces(ds):
    """juliensimon/agent-traces-code-review-pipeline: agent traces"""
    records = []
    for row in ds:
        user_query = row.get("user_query", "")
        message_content = row.get("message_content", "")
        completion = row.get("completion", "")
        reasoning = row.get("reasoning", "")
        event_type = row.get("event_type", "")

        # Only keep assistant responses with actual content
        if event_type != "message_completed" or not message_content:
            continue
        if not user_query:
            continue

        # Build response with reasoning if available
        if reasoning:
            response = f"<think>\n{reasoning}\n</think>\n\n{message_content}"
        else:
            response = message_content

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response},
            ],
            "source": "agent-traces-code-review",
        })
    return records


def deduplicate(records):
    """Remove duplicate user messages across all sources"""
    seen = set()
    unique = []
    for r in records:
        user_content = ""
        for msg in r["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        # Hash the user content for dedup
        h = hashlib.md5(user_content.encode("utf-8", errors="replace")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique


def filter_too_long(records, max_tokens=8000):
    """Filter out examples that are too long for training"""
    filtered = []
    for r in records:
        total_len = sum(len(msg["content"]) for msg in r["messages"])
        # Rough estimate: 1 token ~ 4 chars
        if total_len // 4 <= max_tokens:
            filtered.append(r)
    return filtered


def main():
    print("Loading datasets...")

    # 1. ronantakizawa/github-codereview (334K rows)
    print("  Loading github-codereview...")
    ds_gh = load_dataset("ronantakizawa/github-codereview", split="train", trust_remote_code=True)
    records_gh = normalize_github_codereview(ds_gh)
    print(f"    -> {len(records_gh)} valid records")

    # 2. AmanPriyanshu/reasoning-sft-github-codereview (76.7K rows)
    print("  Loading reasoning-sft-github-codereview...")
    ds_rs = load_dataset("AmanPriyanshu/reasoning-sft-github-codereview", split="train", trust_remote_code=True)
    records_rs = normalize_reasoning_sft(ds_rs)
    print(f"    -> {len(records_rs)} valid records")

    # 3. Dahoas/code-review-instruct-critique-revision
    print("  Loading dahoas/code-review-instruct-critique-revision...")
    try:
        ds_dh = load_dataset("Dahoas/code-review-instruct-critique-revision", split="train", trust_remote_code=True)
        records_dh = normalize_dahoas(ds_dh)
        print(f"    -> {len(records_dh)} valid records")
    except Exception as e:
        print(f"    -> Error loading: {e}")
        records_dh = []

    # 4. juliensimon/agent-traces-code-review-pipeline (2K rows)
    print("  Loading agent-traces-code-review-pipeline...")
    ds_at = load_dataset("juliensimon/agent-traces-code-review-pipeline", split="train", trust_remote_code=True)
    records_at = normalize_agent_traces(ds_at)
    print(f"    -> {len(records_at)} valid records")

    # Merge all
    all_records = records_gh + records_rs + records_dh + records_at
    print(f"\nTotal before dedup: {len(all_records)}")

    # Deduplicate
    print("Deduplicating...")
    all_records = deduplicate(all_records)
    print(f"Total after dedup: {len(all_records)}")

    # Filter too long
    print("Filtering too-long examples...")
    all_records = filter_too_long(all_records)
    print(f"Total after length filter: {len(all_records)}")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_records)

    # Split train/val (95/5)
    split_idx = int(len(all_records) * 0.95)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    print(f"\nTrain: {len(train_records)}, Val: {len(val_records)}")

    # Source distribution
    from collections import Counter
    sources = Counter(r["source"] for r in all_records)
    print(f"\nSource distribution:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")

    # Create datasets
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    # Upload to HF
    print("\nUploading to Hugging Face...")
    dataset_dict = {"train": train_ds, "validation": val_ds}
    dataset_dict["train"].push_to_hub("liodon-ai/gemma4-code-review-instruct")
    dataset_dict["validation"].push_to_hub("liodon-ai/gemma4-code-review-instruct", split="validation")

    print(f"\nDone! Dataset uploaded to: https://huggingface.co/datasets/liodon-ai/gemma4-code-review-instruct")
    print(f"Total examples: {len(all_records)}")


if __name__ == "__main__":
    main()
