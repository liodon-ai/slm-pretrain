"""Evaluate SLM-10M on ArithMark-2.0 benchmark."""
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
model_path = "hf_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32).to(device)
model.eval()

ds = load_dataset("AxiomicLabs/ArithMark-2.0", split="train")

correct = 0
total = len(ds)

with torch.no_grad():
    for i, row in enumerate(ds):
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])

        # Tokenize context
        ctx_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)

        # Compute log-likelihood for each ending
        scores = []
        for ending in endings:
            full = tokenizer.encode(ctx + ending, return_tensors="pt").to(device)
            ending_len = len(tokenizer.encode(ending, add_special_tokens=False))
            logits = model(full).logits[0]
            # Cross-entropy loss on the ending tokens
            loss = torch.nn.functional.cross_entropy(
                logits[-ending_len-1:-1], full[0, -ending_len:]
            )
            scores.append(-loss.item())

        pred = scores.index(max(scores))
        if pred == label:
            correct += 1

        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{total} — acc: {correct/(i+1):.4f}")

acc = correct / total
print(f"\nArithMark-2.0 accuracy: {acc:.4f} ({correct}/{total})")
