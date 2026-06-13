from transformers import PretrainedConfig


class SLMConfig(PretrainedConfig):
    model_type = "slm"

    def __init__(
        self,
        vocab_size: int   = 8192,
        hidden_size: int  = 256,
        num_layers: int   = 12,
        num_q_heads: int  = 8,
        num_kv_heads: int = 2,
        head_dim: int     = 32,
        intermediate: int = 640,
        max_position_embeddings: int = 1024,
        rope_theta: float = 100_000.0,
        norm_eps: float   = 1e-6,
        **kwargs,
    ):
        self.vocab_size              = vocab_size
        self.hidden_size             = hidden_size
        self.num_layers              = num_layers
        self.num_q_heads             = num_q_heads
        self.num_kv_heads            = num_kv_heads
        self.head_dim                = head_dim
        self.intermediate            = intermediate
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta              = rope_theta
        self.norm_eps                = norm_eps
        # Alias expected by transformers internals (DynamicCache, etc.)
        self.num_hidden_layers       = num_layers
        super().__init__(**kwargs)
