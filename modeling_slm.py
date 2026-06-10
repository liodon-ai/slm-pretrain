"""
HuggingFace-compatible wrapper for SLM-10M.

Loaded via:
  AutoModelForCausalLM.from_pretrained("you/slm-10m", trust_remote_code=True)

This file is self-contained so it works when pushed to the HF Hub.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_slm import SLMConfig


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig = x.dtype
        x    = x.float()
        out  = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(orig)


# ── QK-Norm ───────────────────────────────────────────────────────────────────

class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x):
        orig = x.dtype
        x    = x.float()
        out  = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(orig)


# ── RoPE ──────────────────────────────────────────────────────────────────────

def build_rope_cache(max_seq_len, head_dim, theta, device):
    pos   = torch.arange(max_seq_len, device=device)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang   = torch.outer(pos, freqs)
    cos   = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1)
    sin   = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1)
    return cos, sin


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    T   = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ── Attention ─────────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.q_heads  = cfg.num_q_heads
        self.kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.groups   = cfg.num_q_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_q_heads  * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_q_heads * cfg.head_dim, cfg.hidden_size,  bias=False)
        self.q_norm = QKNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = QKNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.q_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


# ── FFN ───────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.up   = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Block ─────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── HF Model ──────────────────────────────────────────────────────────────────

class SLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class      = SLMConfig
    _no_split_modules = ["Block"]

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.embed  = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm   = RMSNorm(config.hidden_size, config.norm_eps)
        self.head   = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # tie lm_head to embedding

        cos, sin = build_rope_cache(
            config.max_position_embeddings, config.head_dim,
            config.rope_theta, device=torch.device("cpu"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        # residual projections rescaled after full init in get_model()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def tie_weights(self, **kwargs):
        # Called by from_pretrained after loading — re-ties head to embed.
        self.head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape
        x   = self.embed(input_ids)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)

        logits = self.head(self.norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
