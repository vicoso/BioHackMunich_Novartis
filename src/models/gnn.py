"""
Graph Neural Network models for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
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
        # Context (per-graph) feature dimension to concatenate post-pooling
        self.context_dim = getattr(config, "context_dim", 0)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(self.node_feature_dim, self.hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # Hidden layers
        for _ in range(self.num_conv_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # MLP for prediction after pooling (+ optional context features)
        mlp_input_dim = self.hidden_dim + (
            self.context_dim if self.context_dim else 0
        )
        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_genes)

        self.dropout = nn.Dropout(self.dropout_rate)

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

        # Graph convolution layers with ReLU and batch norm
        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling (aggregate node features to graph-level)
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
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
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_embeddings(self, data) -> torch.Tensor:
        """
        Get graph-level embeddings before final prediction layer.

        Args:
            data: PyTorch Geometric Data object or Batch object

        Returns:
            Graph embeddings of shape (batch_size, hidden_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution layers
        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
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
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
