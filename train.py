"""
Pretraining script for SLM-10M.

Usage:
  python train.py                           # fresh run
  python train.py --resume checkpoints/step_0010000.pt
  python train.py --config '{"wandb": true, "run_name": "my-run"}'
"""

from __future__ import annotations
import os
import json
import math
import time
import logging
import argparse
from dataclasses import asdict

import torch

from config import TrainConfig
from model import SLM, ModelConfig
from data import make_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-stable-decay (trapezoidal) schedule."""
    if step < cfg.warmup_steps:
        # linear warmup
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    if step < cfg.decay_start_step:
        # stable phase
        return cfg.learning_rate
    # cosine decay tail
    progress = (step - cfg.decay_start_step) / max(1, cfg.total_steps - cfg.decay_start_step)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model: SLM, optimizer, step: int, cfg: TrainConfig) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
    tmp  = path + ".tmp"
    torch.save({
        "step":       step,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "train_cfg":  asdict(cfg),
        "model_cfg":  asdict(model.cfg),
    }, tmp)
    os.replace(tmp, path)
    logger.info("  checkpoint → %s", path)


def load_checkpoint(path: str, model: SLM, optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    logger.info("Resumed from %s  (step %d)", path, step)
    return step


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, val_loader, device: str, pt_dtype, n_steps: int = 50) -> float:
    model.eval()
    ctx   = torch.amp.autocast(device_type=device, dtype=pt_dtype)
    total = 0.0
    it    = iter(val_loader)
    for _ in range(n_steps):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_steps


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, resume: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    device   = cfg.device
    pt_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    logger.info("=== SLM-10M Pretraining ===")
    logger.info("device=%s  dtype=%s  compile=%s  seed=%d", device, cfg.dtype, cfg.compile, cfg.seed)

    # ── model ──────────────────────────────────────────────────────────────
    logger.info("Initializing model...")
    model_cfg = ModelConfig()
    model     = SLM(model_cfg).to(device)
    n_params = model.num_params()
    logger.info("Model created: %d parameters (%.3fM)", n_params, n_params/1e6)

    raw_model = model   # reference before compile for checkpointing
    if cfg.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        logger.info("Model compiled.")

    # ── optimizer ──────────────────────────────────────────────────────────
    # apply weight decay only to 2-D parameters (matrices); skip norms/biases
    decay_params   = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
    logger.info("Optimizer: AdamW (lr=%.1e, wd=%.2f, betas=(%.2f, %.2f))",
                cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2)
    logger.info("Parameters: %d decay, %d no-decay", len(decay_params), len(nodecay_params))
    try:
        optimizer = torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": cfg.weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=True,
        )
    except TypeError:
        # fused not available on this build
        optimizer = torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": cfg.weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
        )

    start_step = 0
    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
        start_step = load_checkpoint(resume, raw_model, optimizer, device)
    else:
        logger.info("Fresh start (no resume)")

    # ── data ───────────────────────────────────────────────────────────────
    logger.info("Loading data from %s (seq_len=%d, batch=%d)", cfg.data_dir, cfg.max_seq_len, cfg.micro_batch_size)
    train_loader = make_dataloader(
        cfg.data_dir, cfg.max_seq_len, cfg.micro_batch_size,
        mix=cfg.data_mix, seed=cfg.seed, split="train",
    )
    val_loader = make_dataloader(
        cfg.data_dir, cfg.max_seq_len, cfg.micro_batch_size,
        mix=cfg.data_mix, seed=cfg.seed + 1, split="val",
    )
    train_iter = iter(train_loader)
    logger.info("Data loaders ready.")

    # ── logging setup ──────────────────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_fh = open(cfg.log_file, "a")

    if cfg.wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))
        logger.info("W&B initialized: project=%s, run=%s", cfg.wandb_project, cfg.run_name)

    n_params = raw_model.num_params()
    logger.info("=" * 62)
    logger.info("  SLM Pretraining")
    logger.info("  parameters:    %.3fM", n_params/1e6)
    logger.info("  total steps:   %d", cfg.total_steps)
    logger.info("  batch tokens:  %d  (grad_accum=%d)", cfg.batch_tokens, cfg.grad_accum_steps)
    logger.info("  device:        %s  dtype=%s  compile=%s", device, cfg.dtype, cfg.compile)
    logger.info("  data:          %s", cfg.data_dir)
    logger.info("=" * 62)

    # ── main loop ──────────────────────────────────────────────────────────
    model.train()
    ctx = torch.amp.autocast(device_type=device, dtype=pt_dtype)
    t_log = time.perf_counter()

    for step in range(start_step, cfg.total_steps):
        # set LR for this step
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Disable z-loss after 31B tokens (GPT-X2 recipe)
        tokens_so_far = step * cfg.batch_tokens
        if tokens_so_far >= 31_000_000_000 and raw_model.cfg.z_loss_weight > 0:
            raw_model.cfg.z_loss_weight = 0.0
            print(f"  z-loss disabled at step {step} ({tokens_so_far/1e9:.1f}B tokens)")

        # ── gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()
            loss_accum += loss.item() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ── stdout + file logging ─────────────────────────────────────────
        if step % cfg.log_every == 0:
            now      = time.perf_counter()
            elapsed  = now - t_log
            tok_s    = cfg.batch_tokens * cfg.log_every / elapsed
            t_log    = now
            tokens   = (step + 1) * cfg.batch_tokens
            pct      = 100.0 * tokens / cfg.total_tokens
            eta_h    = (cfg.total_tokens - tokens) / tok_s / 3600

            row = {
                "step":    step,
                "loss":    round(loss_accum, 4),
                "lr":      round(lr, 7),
                "tok_s":   int(tok_s),
                "tokens":  tokens,
                "pct":     round(pct, 3),
            }
            logger.info(
                "step %d | loss %.4f | lr %.2e | %.1fk tok/s | %.1f%% | ETA %.1fh",
                step, loss_accum, lr, tok_s/1e3, pct, eta_h
            )
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            if cfg.wandb:
                import wandb
                wandb.log(row, step=step)

        # ── validation ────────────────────────────────────────────────────
        if step > 0 and step % cfg.eval_every == 0:
            v_loss = eval_loss(model, val_loader, device, pt_dtype)
            logger.info("  val_loss: %.4f", v_loss)
            log_fh.write(json.dumps({"step": step, "val_loss": round(v_loss, 4)}) + "\n")
            log_fh.flush()
            if cfg.wandb:
                import wandb
                wandb.log({"val_loss": v_loss}, step=step)

        # ── checkpoint ────────────────────────────────────────────────────
        if step > 0 and step % cfg.save_every == 0:
            save_checkpoint(raw_model, optimizer, step, cfg)

    # final save
    save_checkpoint(raw_model, optimizer, cfg.total_steps, cfg)
    log_fh.close()
    if cfg.wandb:
        import wandb
        wandb.finish()
    logger.info("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON string of TrainConfig field overrides, e.g. '{\"wandb\": true}'")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config:
        for k, v in json.loads(args.config).items():
            setattr(cfg, k, v)

    train(cfg, resume=args.resume)
