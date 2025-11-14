#!/usr/bin/env python3
"""
Basic usage example for the BioHackMunich Novartis GNN gene expression prediction package.

This example demonstrates how to:
1. Load and prepare molecular data
2. Configure and train a GNN model
3. Evaluate model performance
4. Make predictions on new molecules
"""

import torch
import numpy as np
from torch_geometric.data import DataLoader

# Import our modular components
from src.config import (
    GNNConfig,
    TrainingConfig,
)
from src.models.gnn import MolecularGCN
from src.training.trainer import (
    create_dataset,
    train_model,
    evaluate,
)
from src.data.molecular_features import smiles_to_graph


def main():
    """Main example function demonstrating the package usage."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =============================================================================
    # 1. Prepare example data
    # =============================================================================

    print("\n1. Preparing example molecular data...")

    # Example SMILES strings (replace with your actual data)
    smiles_list = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
        "CC(C)(C)C",  # Tert-butane
        "CCCCCCCCCCCCCCCC(=O)O",  # Palmitic acid
        "CC(=O)N",  # Acetamide
    ]

    # Generate dummy gene expression data (978 genes for L1000)
    # In practice, replace this with your actual gene expression data
    num_genes = 978
    num_samples = len(smiles_list)

    # Create realistic-looking gene expression changes
    np.random.seed(42)
    gene_expression_list = []

    for i in range(num_samples):
        # Simulate different expression patterns for different molecules
        base_expression = np.random.normal(0, 0.5, num_genes)
        # Add some structure - some genes are more affected
        strong_genes = np.random.choice(num_genes, size=50, replace=False)
        base_expression[strong_genes] += np.random.normal(0, 1.5, 50)
        gene_expression_list.append(base_expression)

    print(
        f"Loaded {len(smiles_list)} molecules with {num_genes} gene expression values each"
    )

    # =============================================================================
    # 2. Create dataset and data loaders
    # =============================================================================

    print("\n2. Creating molecular graphs dataset...")

    dataset = create_dataset(
        smiles_list,
        gene_expression_list,
        use_chirality=True,
        use_stereochemistry=True,
        add_explicit_hydrogens=False,
    )

    print(f"Successfully created {len(dataset)} molecular graphs")

    # Split dataset (simple approach - in practice use proper train/val/test splits)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=2, shuffle=False)
        if val_dataset
        else None
    )

    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}"
    )

    # =============================================================================
    # 3. Configure and initialize model
    # =============================================================================

    print("\n3. Configuring and initializing model...")

    # Create model configuration
    model_config = GNNConfig(
        node_feature_dim=52,  # Based on molecular features
        hidden_dim=64,  # Smaller for this example
        num_conv_layers=2,  # Fewer layers for faster training
        num_genes=num_genes,
        dropout=0.2,
        pooling="mean",
    )

    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=2,
        num_epochs=20,  # Fewer epochs for demonstration
        patience=5,
        device=str(device),
    )

    # Initialize model
    model = MolecularGCN(model_config).to(device)

    print(f"Model initialized with {model.get_num_parameters()} parameters")
    print("Model architecture:")
    print(model)

    # =============================================================================
    # 4. Train the model
    # =============================================================================

    # print("\n4. Training the model...")

    # # Initialize optimizer and loss function
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=training_config.learning_rate
    # )
    # criterion = torch.nn.MSELoss()

    # # Train model
    # training_history = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     device=device,
    #     num_epochs=training_config.num_epochs,
    #     patience=training_config.patience,
    #     save_best_model=True,
    #     model_save_path="best_model.pth",
    #     verbose=True,
    # )

    # print(f"Training completed at epoch {training_history['final_epoch']}")
    # print(
    #     f"Best validation loss: {min(training_history['val_losses']) if training_history['val_losses'] else 'N/A'}"
    # )

    # # =============================================================================
    # # 5. Evaluate model performance
    # # =============================================================================

    # print("\n5. Evaluating model performance...")

    # if val_loader:
    #     val_metrics, val_predictions, val_targets = evaluate(
    #         model, val_loader, criterion, device
    #     )

    #     print("Validation Metrics:")
    #     for metric, value in val_metrics.items():
    #         print(f"  {metric}: {value:.4f}")

    # # =============================================================================
    # # 6. Make predictions on new molecules
    # # =============================================================================

    # print("\n6. Making predictions on new molecules...")

    # # Example new molecules for prediction
    # new_smiles = [
    #     "CC(C)O",  # Isopropanol
    #     "C1=CC=C(C=C1)O",  # Phenol
    # ]

    # model.eval()
    # with torch.no_grad():
    #     for smiles in new_smiles:
    #         try:
    #             # Convert to graph
    #             graph_data = smiles_to_graph(smiles)
    #             graph_data = graph_data.to(device)

    #             # Add batch dimension
    #             graph_data.batch = torch.zeros(
    #                 graph_data.x.size(0), dtype=torch.long, device=device
    #             )

    #             # Make prediction
    #             prediction = model(graph_data)

    #             print(f"\nMolecule: {smiles}")
    #             print(f"  Predicted gene expression shape: {prediction.shape}")
    #             print(
    #                 f"  Sample predictions (first 5 genes): {prediction[0, :5].cpu().numpy()}"
    #             )
    #             print(
    #                 f"  Mean predicted change: {prediction.mean().item():.4f}"
    #             )
    #             print(f"  Std of predictions: {prediction.std().item():.4f}")

    #         except Exception as e:
    #             print(f"Error processing {smiles}: {e}")

    # # =============================================================================
    # # 7. Model analysis
    # # =============================================================================

    # print("\n7. Model analysis...")

    # # Get parameter information
    # param_info = model.get_parameter_info()
    # print(f"Total parameters: {param_info['total_parameters']}")
    # print(f"Trainable parameters: {param_info['trainable_parameters']}")

    # # Example of getting molecular embeddings (features before final prediction)
    # with torch.no_grad():
    #     example_graph = smiles_to_graph("CCO")  # Ethanol
    #     example_graph = example_graph.to(device)
    #     example_graph.batch = torch.zeros(
    #         example_graph.x.size(0), dtype=torch.long, device=device
    #     )

    #     embeddings = model.get_embeddings(example_graph)
    #     print(f"Molecular embedding shape: {embeddings.shape}")
    #     print(f"Embedding sample: {embeddings[0, :5].cpu().numpy()}")

    # print("\n✅ Example completed successfully!")
    # print("\nNext steps:")
    # print(
    #     "1. Replace dummy data with your actual SMILES and gene expression data"
    # )
    # print("2. Implement proper train/validation/test splits")
    # print("3. Tune hyperparameters using validation performance")
    # print("4. Add data preprocessing and augmentation")
    # print("5. Implement cross-validation for robust evaluation")


if __name__ == "__main__":
    main()
