"""
Graph Neural Network models for molecular property prediction.

`MolecularGCN` applies stacked GCNConv layers with configurable activation and
optional BatchNorm, aggregates node features to graph-level via global pooling
("mean", "add", or "max"), then predicts with an MLP.

Context features (`data.mol_features`) handling:
- If `context_dim > 0` and a sample provides `mol_features` (shape `[context_dim]`
    or `[B, context_dim]`), these are concatenated to the pooled graph embedding
    prior to the MLP, enabling conditioning on per-graph metadata (e.g., dose, platform).

Embeddings:
- `get_embeddings(data)` returns graph-level embeddings after pooling and the first
    two MLP layers (thus including any concatenated `mol_features`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from typing import Union

from .base import BaseModel
from ..config import GNNConfig


class MolecularGCN(BaseModel):
    """
    Graph Convolutional Network for molecular property prediction.
    Uses GCN layers followed by global pooling and MLP for prediction.
    """

    def __init__(self, config: Union[GNNConfig, dict]):
        """
        Initialize the MolecularGCN model.

        Args:
            config: Model configuration (GNNConfig object or dict)
        """
        # Convert dict to GNNConfig if needed
        if isinstance(config, dict):
            config = GNNConfig(**config)

        super().__init__(config.__dict__)

        self.node_feature_dim = config.node_feature_dim
        self.hidden_dim = config.hidden_dim
        self.num_conv_layers = config.num_conv_layers
        self.num_genes = config.num_genes
        self.dropout_rate = config.dropout
        self.pooling = config.pooling
        self.activation_name = getattr(config, "activation", "relu").lower()
        self.use_batch_norm = getattr(config, "use_batch_norm", True)
        # Context (per-graph) feature dimension to concatenate post-pooling
        self.context_dim = getattr(config, "context_dim", 0)
        # Edge feature dimension (bond features + curvature)
        self.edge_feature_dim = getattr(config, "edge_feature_dim", 0)
        self.use_edge_features = self.edge_feature_dim > 0
        # L2 decay coefficient applied to trainable weights (excludes biases/norm params)
        self.weight_decay = float(getattr(config, "weight_decay", 0.0))
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Edge feature processing (if available)
        if self.use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # First layer - use GINEConv if edge features available, else GCNConv
        if self.use_edge_features:
            # MLP for node update in GINE
            first_mlp = nn.Sequential(
                nn.Linear(self.node_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.convs.append(GINEConv(first_mlp, edge_dim=self.hidden_dim))
        else:
            self.convs.append(GCNConv(self.node_feature_dim, self.hidden_dim))

        if self.use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # Hidden layers
        for _ in range(self.num_conv_layers - 1):
            if self.use_edge_features:
                # MLP for node update in GINE
                hidden_mlp = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                self.convs.append(
                    GINEConv(hidden_mlp, edge_dim=self.hidden_dim)
                )
            else:
                self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # MLP for prediction after pooling (+ optional context features)
        mlp_input_dim = self.hidden_dim + (
            self.context_dim if self.context_dim else 0
        )
        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_genes)

        self.dropout = nn.Dropout(self.dropout_rate)

        # Activation function
        self._act = self._resolve_activation(self.activation_name)

    @staticmethod
    def _resolve_activation(name: str):
        name = (name or "relu").lower()
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        if name == "elu":
            return F.elu
        if name == "leaky_relu":
            return F.leaky_relu
        if name == "tanh":
            return torch.tanh
        # default fallback
        return F.relu

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object or Batch object
                  Must have attributes: x, edge_index, batch

        Returns:
            Predicted gene expression values (log-fold changes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process edge features if available
        edge_attr = None
        if (
            self.use_edge_features
            and hasattr(data, "edge_attr")
            and data.edge_attr is not None
        ):
            edge_attr = self.edge_encoder(data.edge_attr)

        # Graph convolution layers with activation and optional batch norm
        for i in range(self.num_conv_layers):
            if self.use_edge_features and edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self._act(x)
            x = self.dropout(x)

        # Global pooling (aggregate node features to graph-level)
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Concatenate optional per-graph context features if present
        if (
            self.context_dim
            and hasattr(data, "mol_features")
            and data.mol_features is not None
        ):
            ctx = data.mol_features
            if ctx.dim() == 1:
                ctx = ctx.view(-1, self.context_dim)
            if ctx.size(1) != self.context_dim:
                raise ValueError(
                    f"mol_features width {ctx.size(1)} != context_dim {self.context_dim}"
                )
            ctx = ctx.to(x.dtype)
            x = torch.cat([x, ctx], dim=1)

        # MLP for final prediction
        x = self._act(self.fc1(x))
        x = self.dropout(x)
        x = self._act(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def l2_regularization(self) -> torch.Tensor:
        """Return L2 penalty scaled by weight_decay for regularizing training."""
        if self.weight_decay <= 0:
            return torch.tensor(0.0, device=self.fc1.weight.device)

        penalty = torch.tensor(0.0, device=self.fc1.weight.device)

        for param in self.parameters():
            if not param.requires_grad:
                continue
            # Skip biases and affine norm parameters (1-D tensors)
            if param.dim() <= 1:
                continue
            penalty = penalty + torch.sum(param.pow(2))

        return self.weight_decay * penalty

    def get_embeddings(self, data) -> torch.Tensor:
        """
        Get graph-level embeddings before final prediction layer.

        Args:
            data: PyTorch Geometric Data object or Batch object

        Returns:
            Graph embeddings of shape (batch_size, hidden_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process edge features if available
        edge_attr = None
        if (
            self.use_edge_features
            and hasattr(data, "edge_attr")
            and data.edge_attr is not None
        ):
            edge_attr = self.edge_encoder(data.edge_attr)

        # Graph convolution layers
        for i in range(self.num_conv_layers):
            if self.use_edge_features and edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self._act(x)
            x = self.dropout(x)

        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Concatenate optional per-graph context features if present
        if (
            self.context_dim
            and hasattr(data, "mol_features")
            and data.mol_features is not None
        ):
            ctx = data.mol_features
            if ctx.dim() == 1:
                ctx = ctx.view(-1, self.context_dim)
            if ctx.size(1) != self.context_dim:
                raise ValueError(
                    f"mol_features width {ctx.size(1)} != context_dim {self.context_dim}"
                )
            ctx = ctx.to(x.dtype)
            x = torch.cat([x, ctx], dim=1)

        # Apply first two MLP layers
        x = self._act(self.fc1(x))
        x = self.dropout(x)
        x = self._act(self.fc2(x))

        return x
