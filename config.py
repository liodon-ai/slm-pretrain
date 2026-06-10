from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Paths
    data_dir:        str = "data"
    tokenizer_path:  str = "tokenizer.json"
    checkpoint_dir:  str = "checkpoints"
    log_file:        str = "train_log.jsonl"

    # Data sources and their sampling weights (must sum to 1.0)
    data_mix: dict = field(default_factory=lambda: {
        "fineweb_edu":   0.55,
        "cosmopedia_v2": 0.25,
        "fineweb_hq":    0.10,
        "finemath":      0.10,
    })

    # Token budget
    total_tokens:      int = 75_000_000_000   # 75B
    tokenizer_n_docs:  int = 2_000_000        # docs sampled to train the tokenizer

    # Sequence / batch
    max_seq_len:      int = 1024
    micro_batch_size: int = 32               # sequences per GPU step before accumulation
    batch_tokens:     int = 524_288          # effective tokens per optimizer step (512K)

    # Optimizer
    learning_rate: float = 3e-3
    min_lr:        float = 3e-4
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # LR schedule: warmup → stable → cosine decay
    warmup_steps: int   = 1_000
    decay_frac:   float = 0.15   # last 15% of steps used for cosine decay tail

    # Checkpointing / logging
    save_every: int = 2_000
    log_every:  int = 10
    eval_every: int = 500

    # Compute
    device:  str  = "cuda"
    dtype:   str  = "bfloat16"
    compile: bool = True
    seed:    int  = 42

    # Optional W&B
    wandb:         bool = False
    wandb_project: str  = "slm-pretrain"
    run_name:      str  = "slm-10m-75b"

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_tokens // (self.micro_batch_size * self.max_seq_len)

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def decay_start_step(self) -> int:
        return self.total_steps - int(self.total_steps * self.decay_frac)
