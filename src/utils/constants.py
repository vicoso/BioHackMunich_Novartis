"""
Constants and default parameters for molecular GNN models.
"""

from typing import List
from rdkit import Chem


# Permitted atoms for molecular feature extraction
PERMITTED_ATOMS: List[str] = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]


# Permitted bond types for molecular feature extraction
PERMITTED_BOND_TYPES: List[Chem.rdchem.BondType] = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Feature dimensions
ATOM_FEATURE_DIM = (
    len(PERMITTED_ATOMS) + 7 + 1
)  # atoms + properties + chirality
BOND_FEATURE_DIM = (
    len(PERMITTED_BOND_TYPES) + 2 + 1
)  # bond types + properties + stereochemistry


# Default model parameters
# Use computed ATOM_FEATURE_DIM to avoid drift from feature definition
DEFAULT_MODEL_PARAMS = {
    "node_feature_dim": ATOM_FEATURE_DIM,
    "hidden_dim": 128,
    "num_conv_layers": 3,
    "num_genes": 978,  # L1000 landmark genes
    "dropout": 0.2,
    "pooling": "mean",
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "patience": 10,  # For early stopping
    "min_delta": 1e-4,  # Minimum improvement for early stopping
}

# Default data preprocessing parameters
DEFAULT_DATA_PARAMS = {
    "use_chirality": True,
    "use_stereochemistry": True,
    "add_explicit_hydrogens": False,
    "normalize_features": True,
}

# Feature dimensions
ATOM_FEATURE_DIM = (
    len(PERMITTED_ATOMS) + 7 + 1
)  # atoms + properties + chirality
BOND_FEATURE_DIM = (
    len(PERMITTED_BOND_TYPES) + 2 + 1
)  # bond types + properties + stereochemistry
