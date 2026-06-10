"""
SLM-10M — 9.968M parameter causal LM targeting the Open SLM Leaderboard <10M tier.

Architecture:
  vocab=8192, hidden=256, layers=12, q_heads=8, kv_heads=2,
  head_dim=32, intermediate=640, ctx=1024
  RMSNorm | RoPE (theta=10000) | GQA | SwiGLU | weight-tied embeddings
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int   = 8192
    hidden_size: int  = 256
    num_layers: int   = 12
    num_q_heads: int  = 8
    num_kv_heads: int = 2
    head_dim: int     = 32    # hidden_size // num_q_heads
    intermediate: int = 640
    max_seq_len: int  = 1024
    rope_theta: float = 10_000.0
    norm_eps: float   = 1e-6

    def __post_init__(self):
        assert self.hidden_size == self.num_q_heads * self.head_dim
        assert self.num_q_heads % self.num_kv_heads == 0


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── Rotary Position Embedding ─────────────────────────────────────────────────

def build_rope_cache(max_seq_len: int, head_dim: int, theta: float, device: torch.device):
    positions = torch.arange(max_seq_len, device=device)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(positions, freqs)             # (seq, head_dim//2)
    half_cos, half_sin = torch.cos(angles), torch.sin(angles)
    cos = torch.cat([half_cos, half_cos], dim=-1)      # (seq, head_dim)
    sin = torch.cat([half_sin, half_sin], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, head_dim)
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)            # (1, 1, T, head_dim)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q_heads  = cfg.num_q_heads
        self.kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.groups   = cfg.num_q_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_q_heads  * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_q_heads * cfg.head_dim, cfg.hidden_size,  bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.q_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ── SwiGLU Feed-Forward ───────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.up   = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm   = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.head   = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight          # weight tying

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
                                    device=torch.device('cpu'))
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len

        x      = self.embed(input_ids)
        cos    = self.rope_cos[:T].to(x.device)
        sin    = self.rope_sin[:T].to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x      = self.norm(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        # parameters() deduplicates tied weights, so this is the true unique count
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg   = ModelConfig()
    model = SLM(cfg)
    total = model.num_params()
    print(f"Parameters (excl. tied lm_head): {total:,}  ({total/1e6:.3f}M)")

    ids  = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt  = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    _, loss = model(ids, tgt)
    print(f"Forward pass ok — loss: {loss.item():.4f}  (expected ~{math.log(cfg.vocab_size):.2f})")
