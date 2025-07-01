from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    model_name: str = "mini-gpt"
    maxlen: int = 256
    vocab_size: int = 50304
    embed_dim: int = 8
    num_heads: int = 2
    feed_forward_dim: int = 8
    num_transformer_blocks: int = 1
    dropout_rate: float = 0.1
    dropconnect_rate: float = 0.1
    use_dropconnect: bool = False
    use_softermax: bool = False
    power: float = 1.0
    use_activation: bool = False
    use_yatnmn: bool = False

@dataclass
class DataConfig:
    dataset_name: str = 'roneneldan/TinyStories'
    split: str = 'train'
    validation_split_name: str = 'validation'
    batch_size: int = 256
    tokenizer_name: str = 'gpt2'
    # Data loading configuration
    loader: str = "tf"  # 'grain' or 'tf'
    use_cache: bool = True  # Whether to cache the dataset. Applies to both 'grain' and 'tf' loaders.
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 10_000
    cache_size: int = 10_000  # Used only for the 'grain' loader's cache.
    num_threads: int = 2
    prefetch_buffer_size: int = 50
    tokenization_batch_size: int = 1000
    use_fast_tokenizer: bool = True

@dataclass
class TrainConfig:
    optimizer_name: str = "adam"
    num_epochs: int = 1
    learning_rate: float = 1e-3
    lr_warmup_steps: int = 2000
    lr_num_decay_steps: int = 100000
    weight_decay: float = 1e-4
    grad_clip_value: float = 1.0
    log_interval: int = 1
    eval_interval: int = 100
    text_log_interval: int = 200
    use_wandb: bool = True
    checkpoint_dir: str = 'checkpoints'
    debug: bool = False
    run_generation: bool = True
    log_determinants: bool = False
    log_gradients: bool = False
    log_batch_stats: bool = False
    log_batch_identity: bool = False
    start_prompt: str = "The difference between the mind and the cosmos is"
    loss_function: str = "optax"  # 'optax' or 'softermax'

@dataclass
class ProjectConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig) 