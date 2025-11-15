"""
W&B-tunable training script for MolecularGCN using Lightning.

- Parses CLI args for common hyperparameters
- Merges wandb.config overrides when running under a W&B sweep/agent
- Reuses dataset/dataloader creation from src.main

Example (single run):
  python scripts/tune.py --learning-rate 3e-4 --batch-size 32 --hidden-dim 256

Example (sweep):
  wandb sweep sweeps/molecular_gcn_grid.yaml
  wandb agent <entity>/<project>/<sweep_id>
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Ensure project root is on path so we can import src.* when running this script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config.cls import (
    GNNConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
)
from src.models.gnn import MolecularGCN
from src.training.lightning import LightningWrapper
from src.main import create_dataset_from_csv, create_dataloaders


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"y", "yes", "t", "true", "1", "on"}:
        return True
    if s in {"n", "no", "f", "false", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune MolecularGCN with W&B sweeps")

    # W&B
    p.add_argument("--wandb-project", type=str, default="gnn_gene_expression")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-off", action="store_true", help="Disable W&B logging"
    )

    # Model (GNN)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-conv-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--pooling",
        type=str,
        default="add",
        choices=["mean", "add", "max"],
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "elu", "leaky_relu", "tanh"],
    )
    p.add_argument(
        "--use-batch-norm",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable BatchNorm (accepts true/false)",
    )
    p.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable BatchNorm (overrides)",
    )

    # Training
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)

    # Data
    p.add_argument(
        "--use-chirality",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    p.add_argument("--no-chirality", action="store_true")
    p.add_argument(
        "--use-stereochemistry",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    p.add_argument("--no-stereochemistry", action="store_true")
    p.add_argument(
        "--add-explicit-hydrogens",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
    )

    return p.parse_args()


def merge_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args with wandb.config (if in a sweep). wandb.config takes precedence."""
    base = vars(args).copy()
    if wandb.run is not None:
        for k, v in wandb.config.items():
            base[k.replace("-", "_")] = v
    # Normalize mutually exclusive toggles
    if base.get("no_batch_norm"):
        base["use_batch_norm"] = False
    if base.get("no_chirality"):
        base["use_chirality"] = False
    if base.get("no_stereochemistry"):
        base["use_stereochemistry"] = False
    return base


def build_experiment_config(cfg: Dict[str, Any]) -> ExperimentConfig:
    exp = ExperimentConfig.default()

    # Model
    exp.model.hidden_dim = int(cfg.get("hidden_dim", exp.model.hidden_dim))
    exp.model.num_conv_layers = int(
        cfg.get("num_conv_layers", exp.model.num_conv_layers)
    )
    exp.model.dropout = float(cfg.get("dropout", exp.model.dropout))
    exp.model.pooling = str(cfg.get("pooling", exp.model.pooling))
    exp.model.activation = str(
        cfg.get("activation", getattr(exp.model, "activation", "relu"))
    )
    exp.model.use_batch_norm = bool(
        cfg.get("use_batch_norm", getattr(exp.model, "use_batch_norm", True))
    )

    # Training
    exp.training.learning_rate = float(
        cfg.get("learning_rate", exp.training.learning_rate)
    )
    exp.training.batch_size = int(
        cfg.get("batch_size", exp.training.batch_size)
    )
    exp.training.num_epochs = int(
        cfg.get("num_epochs", exp.training.num_epochs)
    )
    exp.training.patience = int(cfg.get("patience", exp.training.patience))
    exp.training.min_delta = float(cfg.get("min_delta", exp.training.min_delta))

    # Data
    exp.data.use_chirality = bool(
        cfg.get("use_chirality", exp.data.use_chirality)
    )
    exp.data.use_stereochemistry = bool(
        cfg.get("use_stereochemistry", exp.data.use_stereochemistry)
    )
    exp.data.add_explicit_hydrogens = bool(
        cfg.get("add_explicit_hydrogens", exp.data.add_explicit_hydrogens)
    )

    return exp


def main():
    args = parse_args()

    # Start a W&B run unless disabled
    wandb_run = None
    if not args.wandb_off:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # Merge config (wandb overrides CLI when present)
    merged = merge_wandb_config(args)
    exp = build_experiment_config(merged)

    # === Dataset ===
    graph_data = create_dataset_from_csv()
    if len(graph_data) == 0:
        raise ValueError("No valid graphs created from dataset")

    # Determine target width and context_dim dynamically
    y0 = getattr(graph_data[0], "y", None)
    if y0 is None:
        raise ValueError("First dataset item has no 'y' target")
    n_genes = int(y0.numel()) if isinstance(y0, torch.Tensor) else int(len(y0))
    exp.model.num_genes = n_genes

    if (
        hasattr(graph_data[0], "mol_features")
        and graph_data[0].mol_features is not None
    ):
        exp.model.context_dim = int(graph_data[0].mol_features.shape[-1])
    else:
        exp.model.context_dim = 0

    # === Dataloaders ===
    train_loader, val_loader, test_loader = create_dataloaders(
        graph_data, exp.data, exp.training
    )

    # === Model, Optimizer, Scheduler ===
    model = MolecularGCN(exp.model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp.training.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Flatten config for logging
    config_dict = {
        **exp.model.__dict__,
        **exp.training.__dict__,
        **exp.data.__dict__,
        "experiment_name": exp.experiment_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_samples": len(graph_data),
    }

    lightning_model = LightningWrapper(
        gnn_model=model,
        training_config=exp.training,
        optimizer=optimizer,
        scheduler=scheduler,
        config_dict=config_dict,
    )

    # Logger
    logger = None
    if not args.wandb_off:
        logger = WandbLogger(
            project=args.wandb_project,
            name=(args.wandb_run_name or None),
            config=config_dict,
            save_dir="outputs/wandb_logs",
        )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=exp.training.patience,
            min_delta=exp.training.min_delta,
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

    # Trainer
    trainer = Trainer(
        max_epochs=exp.training.num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    # Train & Test
    trainer.fit(lightning_model, train_loader, val_loader)
    trainer.test(lightning_model, test_loader)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
