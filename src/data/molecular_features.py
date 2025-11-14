"""
Molecular feature extraction utilities for converting SMILES to graph representations.
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import List, Optional, Union

from ..utils.constants import PERMITTED_ATOMS, PERMITTED_BOND_TYPES


def get_atom_features(
    atom: Chem.rdchem.Atom, use_chirality: bool = True
) -> List[Union[int, float]]:
    """
    Convert RDKit atom object to feature vector.

    Args:
        atom: RDKit atom object
        use_chirality: Whether to include chirality information

    Returns:
        List of atom features
    """
    # Atom type (one-hot encoding)
    atom_type = atom.GetSymbol()
    atom_type_enc = [int(atom_type == x) for x in PERMITTED_ATOMS]

    # Atom properties
    features = atom_type_enc + [
        atom.GetTotalNumHs(includeNeighbors=True)
        / 8.0,  # Number of hydrogens (normalized)
        atom.GetTotalDegree() / 4.0,  # Degree (normalized)
        atom.GetFormalCharge(),  # Formal charge
        int(atom.GetIsAromatic()),  # Is aromatic
        atom.GetTotalValence() / 6.0,  # Valence (normalized)
        atom.GetNumRadicalElectrons(),  # Number of radical electrons
        int(atom.IsInRing()),  # Is in ring
    ]

    if use_chirality:
        try:
            chiral_type = int(atom.GetChiralTag())
        except:
            chiral_type = 0
        features.append(chiral_type)

    return features


def get_bond_features(
    bond: Chem.rdchem.Bond, use_stereochemistry: bool = True
) -> List[Union[int, float]]:
    """
    Convert RDKit bond object to feature vector.

    Args:
        bond: RDKit bond object
        use_stereochemistry: Whether to include stereochemistry

    Returns:
        List of bond features
    """
    bond_type = bond.GetBondType()
    bond_type_enc = [int(bond_type == x) for x in PERMITTED_BOND_TYPES]

    features = bond_type_enc + [
        int(bond.GetIsConjugated()),  # Is conjugated
        int(bond.IsInRing()),  # Is in ring
    ]

    if use_stereochemistry:
        try:
            stereo = int(bond.GetStereo())
        except:
            stereo = 0
        features.append(stereo)

    return features


def smiles_to_graph(
    smiles: str,
    use_chirality: bool = True,
    use_stereochemistry: bool = True,
    add_explicit_hydrogens: bool = False,
) -> Data:
    """
    Convert SMILES string to PyTorch Geometric Data object.

    Args:
        smiles: SMILES string representation of molecule
        use_chirality: Whether to include chirality information
        use_stereochemistry: Whether to include stereochemistry information
        add_explicit_hydrogens: Whether to add explicit hydrogens

    Returns:
        PyTorch Geometric Data object with node features, edge indices, and edge features

    Raises:
        ValueError: If SMILES string is invalid
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add explicit hydrogens if requested
    if add_explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Get node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(
            get_atom_features(atom, use_chirality=use_chirality)
        )

    x = torch.tensor(node_features, dtype=torch.float)

    # Get edge indices and edge features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = get_bond_features(
            bond, use_stereochemistry=use_stereochemistry
        )

        # Add both directions (undirected graph represented as bidirectional)
        edge_indices.append([i, j])
        edge_features.append(bond_feat)

        edge_indices.append([j, i])
        edge_features.append(bond_feat)

    # Handle molecules with no bonds (single atoms)
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        # Use default bond feature dimension
        bond_feat_dim = (
            len(PERMITTED_BOND_TYPES) + 2 + (1 if use_stereochemistry else 0)
        )
        edge_attr = torch.zeros((0, bond_feat_dim), dtype=torch.float)
    else:
        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        )
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is chemically valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_molecular_descriptors(smiles: str) -> dict:
    """
    Calculate common molecular descriptors from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of molecular descriptors

    Raises:
        ValueError: If SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    from rdkit.Chem import Descriptors

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": Descriptors.RingCount(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
    }
