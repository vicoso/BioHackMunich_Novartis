"""
Training utilities and helper functions.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional


def calculate_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression tasks.

    Args:
        predictions: Model predictions (batch_size, num_genes)
        targets: Ground truth values (batch_size, num_genes)

    Returns:
        Dictionary of calculated metrics
    """
    # Convert to numpy arrays
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()

    # Calculate metrics
    mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
    mae = mean_absolute_error(target_np.flatten(), pred_np.flatten())
    rmse = np.sqrt(mse)

    # Calculate R2 score (coefficient of determination)
    try:
        r2 = r2_score(target_np.flatten(), pred_np.flatten())
    except:
        r2 = 0.0

    # Calculate per-gene metrics (average across samples)
    gene_mse = np.mean((pred_np - target_np) ** 2, axis=0)
    gene_mae = np.mean(np.abs(pred_np - target_np), axis=0)

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "gene_mse_mean": np.mean(gene_mse),
        "gene_mse_std": np.std(gene_mse),
        "gene_mae_mean": np.mean(gene_mae),
        "gene_mae_std": np.std(gene_mae),
    }

    return metrics


def pearson_correlation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """
    Calculate Pearson correlation coefficient between predictions and targets.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Pearson correlation coefficient
    """
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)

    # Calculate means
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)

    # Calculate correlation
    numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean))
    pred_var = torch.sum((pred_flat - pred_mean) ** 2)
    target_var = torch.sum((target_flat - target_mean) ** 2)

    denominator = torch.sqrt(pred_var * target_var)

    if denominator == 0:
        return 0.0

    correlation = numerator / denominator
    return correlation.item()


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False


class LossTracker:
    """
    Utility to track and log training losses and metrics.
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def update(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Update tracked losses and metrics.

        Args:
            train_loss: Training loss for current epoch
            val_loss: Validation loss for current epoch
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)

        if train_metrics is not None:
            self.train_metrics.append(train_metrics)

        if val_metrics is not None:
            self.val_metrics.append(val_metrics)

    def get_best_epoch(self) -> int:
        """Get epoch with best validation loss."""
        if not self.val_losses:
            return len(self.train_losses) - 1
        return int(np.argmin(self.val_losses))

    def print_epoch_summary(self, epoch: int):
        """Print summary for current epoch."""
        print(f"Epoch {epoch + 1}")
        print(f"  Train Loss: {self.train_losses[-1]:.4f}")

        if self.val_losses:
            print(f"  Val Loss: {self.val_losses[-1]:.4f}")

        if self.val_metrics:
            metrics = self.val_metrics[-1]
            print(f"  Val MAE: {metrics.get('mae', 0):.4f}")
            print(f"  Val R2: {metrics.get('r2', 0):.4f}")

    def save_history(self, filepath: str):
        """Save training history to file."""
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }
        torch.save(history, filepath)
