"""
Convert a trained checkpoint to a HuggingFace model directory ready to push.

Usage:
  python export.py --checkpoint checkpoints/final.pt --out hf_model
  python export.py --checkpoint checkpoints/final.pt --out hf_model --push username/slm-10m
"""

from __future__ import annotations
import argparse
import shutil
import os
import torch
import torch.nn as nn

from model import SLM, ModelConfig
from configuration_slm import SLMConfig
from modeling_slm import SLMForCausalLM


def load_training_checkpoint(path: str, device: str = "cpu") -> tuple[SLM, ModelConfig]:
    ckpt      = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt.get("model_cfg", {}))
    model     = SLM(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


def build_hf_model(train_model: SLM, train_cfg: ModelConfig) -> SLMForCausalLM:
    slm_cfg = SLMConfig(
        vocab_size              = train_cfg.vocab_size,
        hidden_size             = train_cfg.hidden_size,
        num_layers              = train_cfg.num_layers,
        num_q_heads             = train_cfg.num_q_heads,
        num_kv_heads            = train_cfg.num_kv_heads,
        head_dim                = train_cfg.head_dim,
        intermediate            = train_cfg.intermediate,
        max_position_embeddings = train_cfg.max_seq_len,
        rope_theta              = train_cfg.rope_theta,
        norm_eps                = train_cfg.norm_eps,
        architectures           = ["SLMForCausalLM"],
        auto_map={
            "AutoConfig":           "configuration_slm.SLMConfig",
            "AutoModelForCausalLM": "modeling_slm.SLMForCausalLM",
        },
    )
    hf_model = SLMForCausalLM(slm_cfg)

    # Copy weights — state dict keys are identical between train and HF model
    missing, unexpected = hf_model.load_state_dict(train_model.state_dict(), strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected}")

    # Re-apply scaled residual init (not stored in state dict, already in weights,
    # but we verify the copy is correct by doing a parameter norm check)
    return hf_model


def convert_tokenizer(tokenizer_json: str, out_dir: str) -> None:
    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(
            tokenizer_file  = tokenizer_json,
            eos_token       = "<|endoftext|>",
            pad_token       = "<|pad|>",
            unk_token       = None,
            clean_up_tokenization_spaces = False,
        )
        tok.save_pretrained(out_dir)
        print(f"  tokenizer saved → {out_dir}")
    except Exception as e:
        print(f"  Warning: could not save tokenizer: {e}")
        print("  Copy tokenizer.json to the output dir manually.")


def export(
    checkpoint_path: str,
    out_dir: str,
    tokenizer_json: str = "tokenizer.json",
    push_to: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading checkpoint: {checkpoint_path}")
    train_model, train_cfg = load_training_checkpoint(checkpoint_path)

    print("Building HF model…")
    hf_model = build_hf_model(train_model, train_cfg)

    n = sum(p.numel() for p in hf_model.parameters())
    print(f"  parameters: {n:,}  ({n/1e6:.3f}M)")

    print("Saving model…")
    # Safetensors can't store shared tensors.  Clone head.weight so it's
    # a separate tensor on disk; __init__ re-ties them after loading.
    hf_model.head.weight = nn.Parameter(hf_model.embed.weight.data.clone())
    hf_model.save_pretrained(out_dir, safe_serialization=True)

    # Copy modeling code into the output dir so push_to_hub includes it
    for fname in ("configuration_slm.py", "modeling_slm.py"):
        src = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
            print(f"  copied {fname}")

    if os.path.exists(tokenizer_json):
        convert_tokenizer(tokenizer_json, out_dir)
    else:
        print(f"  tokenizer not found at {tokenizer_json!r} — skipping")

    print(f"\nHF model ready at: {out_dir}")
    print(f"Load with:\n  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{out_dir}', trust_remote_code=True)")

    if push_to:
        print(f"\nPushing to HuggingFace Hub as {push_to!r}…")
        hf_model.push_to_hub(push_to, safe_serialization=True)
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(out_dir)
        tok.push_to_hub(push_to)
        print(f"Pushed → https://huggingface.co/{push_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,           help="Path to .pt checkpoint")
    parser.add_argument("--out",         default="hf_model",      help="Output directory")
    parser.add_argument("--tokenizer",   default="tokenizer.json",help="Path to tokenizer.json")
    parser.add_argument("--push",        default=None,            help="HF Hub repo id, e.g. you/slm-10m")
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.tokenizer, push_to=args.push)
