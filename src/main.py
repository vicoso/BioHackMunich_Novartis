"""
Main training script demonstrating PyTorch Lightning wrapper with W&B integration.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any

# Local imports
from .models.gnn import MolecularGCN
from .config.cls import GNNConfig, TrainingConfig, DataConfig, ExperimentConfig
from .data.molecular_features import create_dataset
from .training.lightning import LightningWrapper


def create_dataset_from_csv() -> List[Data]:
    """
    Create PyTorch Geometric dataset from CSV files using helper in molecular_features.

    Returns:
        List of PyTorch Geometric Data objects
    """
    # Load the data files
    data_dir = "data"
    smiles_file = os.path.join(data_dir, "MCE_4000_X.csv")
    targets_file = os.path.join(data_dir, "MCE_4000_Y.csv")

    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")
    if not os.path.exists(targets_file):
        raise FileNotFoundError(f"Targets file not found: {targets_file}")

    # Load X (SMILES + molecular features: pert_idose, seq_platform)
    smiles_df = pd.read_csv(smiles_file)[:200]
    print(
        f"Loaded X with {len(smiles_df)} rows and columns {list(smiles_df.columns)}"
    )

    # Load targets (gene expression data)
    targets_df = pd.read_csv(targets_file)
    print(f"Loaded targets with shape: {targets_df.shape}")

    # Determine counts and align rows across files
    n_samples = min(len(smiles_df), len(targets_df))

    # Extract SMILES list (column 'smiles')
    if "smiles" not in smiles_df.columns:
        raise ValueError("Expected column 'smiles' in MCE_4000_X.csv")
    smiles_list: List[str] = (
        smiles_df.loc[: n_samples - 1, "smiles"].astype(str).tolist()
    )

    # Build molecular features: dose and binary platform
    if (
        "pert_idose" not in smiles_df.columns
        or "seq_platform" not in smiles_df.columns
    ):
        raise ValueError(
            "Expected columns 'pert_idose' and 'seq_platform' in MCE_4000_X.csv"
        )
    pert_vals = smiles_df.loc[: n_samples - 1, "pert_idose"].to_numpy(
        dtype=np.float32
    )
    platform_raw = smiles_df.loc[: n_samples - 1, "seq_platform"]
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(platform_raw):
        platform_vals = platform_raw.to_numpy(dtype=np.float32)
        uniq = np.unique(platform_vals)
        if len(uniq) > 2:
            raise ValueError(
                f"seq_platform expected binary, got unique values: {uniq}"
            )
        if not np.all(np.isin(uniq, [0.0, 1.0])) and len(uniq) == 2:
            minv, maxv = np.min(platform_vals), np.max(platform_vals)
            platform_vals = (platform_vals > minv).astype(np.float32)
    else:
        labels, _ = pd.factorize(platform_raw.astype(str), sort=True)
        if len(np.unique(labels)) > 2:
            raise ValueError(
                f"seq_platform expected binary categories, got: {np.unique(labels)}"
            )
        platform_vals = labels.astype(np.float32)

    mol_feat_matrix = np.stack([pert_vals, platform_vals], axis=1)
    molecular_features_list: List[np.ndarray] = [row for row in mol_feat_matrix]

    # Extract gene expression targets as a list of arrays (one per sample)
    targets_matrix = targets_df.iloc[:n_samples].to_numpy(dtype=np.float32)
    gene_expression_list: List[np.ndarray] = [row for row in targets_matrix]

    print(f"Using {n_samples} samples for training")

    # Build dataset using the shared helper
    graph_data = create_dataset(
        smiles_list=smiles_list,
        molecular_features=molecular_features_list,
        gene_expression_list=gene_expression_list,
        use_chirality=True,
        use_stereochemistry=True,
    )

    print(f"Successfully created dataset with {len(graph_data)} graphs")

    return graph_data


def create_dataloaders(
    graph_data: List[Data],
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> Tuple[Any, Any, Any]:
    """
    Split dataset and create DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split indices
    n_samples = len(graph_data)
    indices = np.arange(n_samples)

    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=data_config.test_split,
        random_state=data_config.random_seed,
        shuffle=True,
    )

    # Second split: separate train and validation
    val_size = data_config.val_split / (
        data_config.train_split + data_config.val_split
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=data_config.random_seed,
        shuffle=True,
    )

    print(
        f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Create datasets
    train_data = [graph_data[i] for i in train_indices]
    val_data = [graph_data[i] for i in val_indices]
    test_data = [graph_data[i] for i in test_indices]

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader


def main():
    """Main training function."""

    # 1. Setup configurations
    config = ExperimentConfig.default()

    # Override some defaults for this example
    config.training.num_epochs = 20
    config.training.batch_size = 16
    config.training.learning_rate = 0.001

    print("Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Training: {config.training}")
    print(f"  Data: {config.data}")

    # 2. Create dataset
    print("\n=== Loading Dataset ===")
    graph_data = create_dataset_from_csv()

    if len(graph_data) == 0:
        raise ValueError("No valid graphs created from dataset")

    # Ensure model output dimension matches Y width
    y0 = getattr(graph_data[0], "y", None)
    if y0 is None:
        raise ValueError("First dataset item has no 'y' target")
    n_genes = int(y0.numel()) if isinstance(y0, torch.Tensor) else int(len(y0))
    # Sanity check: all samples should have same target width
    for i, d in enumerate(graph_data[:50]):  # sample first 50 for speed
        yi = getattr(d, "y", None)
        if yi is None:
            raise ValueError(f"Dataset item {i} missing 'y' target")
        yi_size = (
            int(yi.numel()) if isinstance(yi, torch.Tensor) else int(len(yi))
        )
        if yi_size != n_genes:
            raise ValueError(
                f"Inconsistent target width at index {i}: {yi_size} != {n_genes}"
            )

    # Update model config dynamically based on dataset
    config.model.num_genes = n_genes

    # 3. Create data loaders
    print("\n=== Creating DataLoaders ===")
    train_loader, val_loader, test_loader = create_dataloaders(
        graph_data, config.data, config.training
    )

    # 4. Create model
    print("\n=== Creating Model ===")
    # Set context_dim dynamically from dataset (if mol_features present)
    if (
        hasattr(graph_data[0], "mol_features")
        and graph_data[0].mol_features is not None
    ):
        config.model.context_dim = int(graph_data[0].mol_features.shape[-1])
    else:
        config.model.context_dim = 0
    model = MolecularGCN(config.model)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # 5. Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 6. Create Lightning wrapper
    print("\n=== Creating Lightning Wrapper ===")

    # Flatten config for W&B logging
    config_dict = {
        **config.model.__dict__,
        **config.training.__dict__,
        **config.data.__dict__,
        "experiment_name": config.experiment_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_samples": len(graph_data),
    }

    lightning_model = LightningWrapper(
        gnn_model=model,
        training_config=config.training,
        optimizer=optimizer,
        scheduler=scheduler,
        config_dict=config_dict,
    )

    # 7. Setup W&B logger
    print("\n=== Setting up W&B Logger ===")
    wandb_logger = WandbLogger(
        project="gnn_gene_expression",
        name=f"molecular_gcn_{config.training.batch_size}bs_{config.training.learning_rate}lr",
        config=config_dict,
        save_dir="outputs/wandb_logs",
    )

    # 8. Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="outputs/checkpoints",
            filename="best_model_{epoch:02d}_{val_loss:.3f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        ),
    ]

    # 9. Create trainer
    print("\n=== Creating Trainer ===")
    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    # 10. Train the model
    print("\n=== Starting Training ===")
    trainer.fit(lightning_model, train_loader, val_loader)

    # 11. Test the model
    print("\n=== Testing Model ===")
    trainer.test(lightning_model, test_loader)

    print("\n=== Training Complete ===")
    print(f"Best model saved to: outputs/checkpoints/")
    print(f"W&B logs available at: https://wandb.ai")


if __name__ == "__main__":
    main()
