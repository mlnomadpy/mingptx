from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    maxlen: int = 256
    vocab_size: int = 50257  # GPT-2 vocab size
    embed_dim: int = 256
    num_heads: int = 8
    feed_forward_dim: int = 256
    num_transformer_blocks: int = 8
    dropout_rate: float = 0.1

@dataclass
class DataConfig:
    dataset_name: str = 'roneneldan/TinyStories'
    split: str = 'train'
    batch_size: int = 256
    tokenizer_name: str = 'gpt2'

@dataclass
class TrainConfig:
    num_epochs: int = 1
    learning_rate: float = 1e-3
    log_interval: int = 1
    text_log_interval: int = 200
    use_wandb: bool = True
    checkpoint_dir: str = 'checkpoints'
    
@dataclass
class ProjectConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig) 