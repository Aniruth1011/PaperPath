from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    dataset_name: str = "ogbn-arxiv"
    batch_size: int = 1024
    num_neighbors: Tuple[int, ...] = (15, 10)
    num_workers: int = 4
    pin_memory: bool = True
    make_undirected: bool = True
    normalize_features: bool = True


@dataclass
class ModelConfig:
    in_channels: int = 896       # 128 (OGB node feat) + 768 (SPECTER2)
    hidden_channels: int = 256
    out_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    neg_ratio: int = 3                  # random negatives per positive (fallback only)
    checkpoint_dir: str = "./outputs/checkpoints"
    embeddings_output: str = "./outputs/embeddings/gnn_embeddings.pt"
    log_interval: int = 10

    # Hard negative sampling
    use_hard_negatives: bool = True
    hard_neg_per_node: int = 2          # same-category negatives per seed node per batch

    # Co-citation contrastive loss
    cocitation_weight: float = 0.3      # weight relative to link-prediction loss
    cocitation_temperature: float = 0.07


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()
