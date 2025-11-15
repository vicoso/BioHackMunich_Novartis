"""
Training and evaluation functions for molecular GNN models.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ..data.molecular_features import smiles_to_graph
from .utils import calculate_metrics, EarlyStopping, LossTracker

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping as PL_EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


def create_dataset(
    smiles_list: List[str],
    gene_expression_list: List[np.ndarray],
    use_chirality: bool = True,
    use_stereochemistry: bool = True,
    add_explicit_hydrogens: bool = False,
) -> List[Data]:
    """
    Create dataset from SMILES strings and gene expression values.

    Args:
        smiles_list: List of SMILES strings
        gene_expression_list: List of gene expression arrays (log-fold changes)
        use_chirality: Whether to include chirality information
        use_stereochemistry: Whether to include stereochemistry information
        add_explicit_hydrogens: Whether to add explicit hydrogens

    Returns:
        List of PyTorch Geometric Data objects
    """
    dataset = []

    for smiles, gene_expr in zip(smiles_list, gene_expression_list):
        try:
            # Convert SMILES to graph
            graph_data = smiles_to_graph(
                smiles,
                use_chirality=use_chirality,
                use_stereochemistry=use_stereochemistry,
                add_explicit_hydrogens=add_explicit_hydrogens,
            )

            # Add gene expression as target (shape [1, num_genes] so batches stack to [B, num_genes])
            graph_data.y = torch.tensor(gene_expr, dtype=torch.float).unsqueeze(
                0
            )

            dataset.append(graph_data)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue

    return dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    return_metrics: bool = False,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: GNN model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        return_metrics: Whether to calculate and return additional metrics

    Returns:
        Dictionary with training loss and optionally other metrics
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        pred = model(batch)

        # Compute loss (both should be [B, G])
        loss = criterion(pred, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        if return_metrics:
            all_predictions.append(pred.detach().cpu())
            all_targets.append(batch.y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)

    result = {"loss": avg_loss}

    if return_metrics and all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(predictions, targets)
        result.update(metrics)

    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Evaluate model on validation/test set.

    Args:
        model: GNN model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (metrics dict, predictions tensor, targets tensor)
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            loss = criterion(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs

            predictions.append(pred.cpu())
            targets.append(batch.y.cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    avg_loss = total_loss / len(loader.dataset)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(predictions, targets)
    metrics["loss"] = avg_loss

    return metrics, predictions, targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    save_best_model: bool = True,
    model_save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete training loop with early stopping and model checkpointing.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        save_best_model: Whether to save best model
        model_save_path: Path to save best model
        verbose: Whether to print training progress

    Returns:
        Dictionary with training history and results
    """
    # Initialize tracking utilities
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    loss_tracker = LossTracker()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            return_metrics=True,
        )

        # Validation
        val_result = None
        if val_loader is not None:
            val_result, _, _ = evaluate(model, val_loader, criterion, device)

            # Update loss tracker
            loss_tracker.update(
                train_loss=train_result["loss"],
                val_loss=val_result["loss"],
                train_metrics=train_result,
                val_metrics=val_result,
            )

            # Check for best model
            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                if save_best_model and model_save_path:
                    torch.save(model.state_dict(), model_save_path)

            # Check early stopping
            if early_stopping(val_result["loss"], model):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            loss_tracker.update(
                train_loss=train_result["loss"], train_metrics=train_result
            )

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            loss_tracker.print_epoch_summary(epoch)

    # Return training history
    return {
        "train_losses": loss_tracker.train_losses,
        "val_losses": loss_tracker.val_losses,
        "train_metrics": loss_tracker.train_metrics,
        "val_metrics": loss_tracker.val_metrics,
        "best_epoch": loss_tracker.get_best_epoch(),
        "final_epoch": epoch + 1,
    }
