"""
Configuration classes for model, training, and data parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GNNConfig:
    """Configuration for GNN model architecture."""

    # Matches get_atom_features dimensionality: len(PERMITTED_ATOMS) + 7 properties + (1 if chirality)
    # With current defaults, this evaluates to 51.
    node_feature_dim: int = 51
    hidden_dim: int = 128
    num_conv_layers: int = 3
    num_genes: int = 978
    dropout: float = 0.2
    pooling: str = "add"
    # Nonlinearity used after GCN/linear layers
    activation: str = "relu"
    # Whether to apply BatchNorm1d after graph conv layers
    use_batch_norm: bool = True
    # Number of per-graph context features concatenated before MLP (e.g., dose, platform)
    context_dim: int = 0
    # Number of edge features (bond features + curvature)
    edge_feature_dim: int = 0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pooling not in ["mean", "add", "max"]:
            raise ValueError(
                f"Pooling method must be 'mean', 'add', or 'max', got {self.pooling}"
            )
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(
                f"Dropout must be between 0 and 1, got {self.dropout}"
            )
        if self.num_conv_layers < 1:
            raise ValueError(
                f"Number of conv layers must be >= 1, got {self.num_conv_layers}"
            )
        if self.activation.lower() not in {
            "relu",
            "gelu",
            "elu",
            "leaky_relu",
            "tanh",
        }:
            raise ValueError(
                f"Unsupported activation: {self.activation}. Choose from 'relu','gelu','elu','leaky_relu','tanh'"
            )


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    device: str = "auto"  # "auto", "cpu", or "cuda"
    save_best_model: bool = True
    model_save_path: Optional[str] = None

    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be > 0, got {self.learning_rate}"
            )
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1:
            raise ValueError(
                f"Number of epochs must be >= 1, got {self.num_epochs}"
            )


@dataclass
class DataConfig:
    """Configuration for data preprocessing parameters."""

    use_chirality: bool = True
    use_stereochemistry: bool = True
    add_explicit_hydrogens: bool = False
    use_geometric_features: bool = True
    normalize_features: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    def __post_init__(self):
        """Validate data configuration."""
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Train/val/test splits must sum to 1.0, got {total_split}"
            )
        if any(
            split < 0
            for split in [self.train_split, self.val_split, self.test_split]
        ):
            raise ValueError("All splits must be non-negative")


@dataclass
class ExperimentConfig:
    """Combined configuration for complete experiments."""

    model: GNNConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = "gnn_gene_expression"
    output_dir: str = "outputs"

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Create default experiment configuration."""
        return cls(
            model=GNNConfig(), training=TrainingConfig(), data=DataConfig()
        )
