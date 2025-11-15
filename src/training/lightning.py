"""
PyTorch Lightning wrapper for molecular GNN models with W&B integration.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
from typing import Optional, Dict, Any
from pytorch_lightning.loggers import WandbLogger

from ..config.cls import TrainingConfig


class LightningWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for GNN models with integrated W&B logging.

    Features:
    - RMSE, MAE, and MSE loss metrics
    - Weights & Biases integration
    - Configurable optimizer and scheduler
    - Proper PyTorch Geometric batch handling
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        training_config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Lightning wrapper.

        Args:
            gnn_model: The GNN model to wrap
            training_config: Training configuration object
            optimizer: Optional pre-configured optimizer
            scheduler: Optional pre-configured scheduler
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_entity: W&B entity/team name
            config_dict: Configuration dictionary to log to W&B
        """
        super().__init__()

        # Store model (renamed from gnn_model to backbone for consistency)
        self.backbone = gnn_model
        self.training_config = training_config or TrainingConfig()

        # Store optimizer and scheduler for configure_optimizers
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Loss function - MSE as primary loss
        self.loss_fn = nn.MSELoss()

        # Metrics for training
        self.train_mae = tm.MeanAbsoluteError()
        self.train_rmse = tm.MeanSquaredError(squared=False)  # RMSE
        self.train_mse = tm.MeanSquaredError(squared=True)  # MSE

        # Metrics for validation
        self.val_mae = tm.MeanAbsoluteError()
        self.val_rmse = tm.MeanSquaredError(squared=False)
        self.val_mse = tm.MeanSquaredError(squared=True)

        # Metrics for testing
        self.test_mae = tm.MeanAbsoluteError()
        self.test_rmse = tm.MeanSquaredError(squared=False)
        self.test_mse = tm.MeanSquaredError(squared=True)

        # Save hyperparameters for W&B logging
        if config_dict:
            self.save_hyperparameters(config_dict)
        else:
            # Save training config as hyperparameters
            self.save_hyperparameters(self.training_config.__dict__)

    def step(
        self,
        batch,
        batch_idx: int,
        prefix: str,
        mae_metric,
        rmse_metric,
        mse_metric,
    ):
        """
        Common step function for train/val/test.

        Args:
            batch: PyTorch Geometric batch object
            batch_idx: Batch index
            prefix: Logging prefix ('train', 'val', 'test')
            mae_metric: MAE metric for this split
            rmse_metric: RMSE metric for this split
            mse_metric: MSE metric for this split
        """
        # Get targets from PyG batch (not dict)
        y = batch.y.float()

        # Forward pass through model
        y_pred = self.backbone(batch).float()

        # --- Runtime sanity checks: detect NaN / Inf early and surface informative errors
        def _check(name, tensor):
            if tensor is None:
                return
            if torch.isnan(tensor).any():
                raise ValueError(
                    f"NaN detected in {name} (batch_idx={batch_idx}, shape={tuple(tensor.shape)})"
                )
            if torch.isinf(tensor).any():
                raise ValueError(
                    f"Inf detected in {name} (batch_idx={batch_idx}, shape={tuple(tensor.shape)})"
                )

        # Check common tensors that can cause NaN propagation
        _check("y", y)
        _check("y_pred", y_pred)
        _check("batch.x", getattr(batch, "x", None))
        _check("batch.edge_attr", getattr(batch, "edge_attr", None))
        _check("batch.mol_features", getattr(batch, "mol_features", None))

        # Handle batched data: y should be [batch_size, num_genes], y_pred is [batch_size, num_genes]
        # If y is flattened from batching, reshape it
        if y.dim() == 1 and y_pred.dim() == 2:
            batch_size = y_pred.size(0)
            num_genes = y_pred.size(1)
            if y.size(0) == batch_size * num_genes:
                y = y.view(batch_size, num_genes)

        # Calculate loss
        loss = self.loss_fn(y_pred, y)

        # Update metrics
        mae_metric(y_pred, y)
        rmse_metric(y_pred, y)
        mse_metric(y_pred, y)

        # Log loss
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
            prog_bar=prefix == "train",
        )

        # Log metrics
        self.log(
            f"{prefix}_mae",
            mae_metric,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{prefix}_rmse",
            rmse_metric,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{prefix}_mse",
            mse_metric,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def training_step(self, batch, batch_idx: int):
        """Training step."""
        return self.step(
            batch,
            batch_idx,
            "train",
            self.train_mae,
            self.train_rmse,
            self.train_mse,
        )

    def validation_step(self, batch, batch_idx: int):
        """Validation step."""
        return self.step(
            batch, batch_idx, "val", self.val_mae, self.val_rmse, self.val_mse
        )

    def test_step(self, batch, batch_idx: int):
        """Test step."""
        return self.step(
            batch,
            batch_idx,
            "test",
            self.test_mae,
            self.test_rmse,
            self.test_mse,
        )

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self._optimizer is not None:
            optimizer = self._optimizer
        else:
            # Default to Adam optimizer
            optimizer = torch.optim.Adam(
                self.backbone.parameters(),
                lr=self.training_config.learning_rate,
            )

        if self._scheduler is not None:
            scheduler = {
                "scheduler": self._scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def on_train_start(self):
        """Called when training starts - log model summary."""
        if isinstance(self.logger, WandbLogger):
            # Log model summary to W&B
            self.logger.watch(self.backbone, log="all", log_freq=100)
