import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import GetMolFrags
from typing import List, Optional, Union, Sequence

from ..utils.constants import PERMITTED_ATOMS, PERMITTED_BOND_TYPES

# --- Feature Helpers ---


def get_atom_features(
    atom: Chem.rdchem.Atom, use_chirality: bool = True
) -> List[float]:
    """
    Convert RDKit atom object to feature vector.
    Features are normalized for consistent scaling in a GNN.
    """
    # 1. Atom type (One-hot encoding)
    atom_type = atom.GetSymbol()
    atom_type_enc = [float(atom_type == x) for x in PERMITTED_ATOMS]

    # 2. Basic properties (Normalized or numeric)
    features = atom_type_enc + [
        float(atom.GetTotalNumHs(includeNeighbors=True))
        / 8.0,  # H-count (normalized)
        float(atom.GetTotalDegree()) / 4.0,  # Degree (normalized)
        float(atom.GetFormalCharge()),  # Formal charge (numeric)
        float(atom.GetIsAromatic()),  # Is aromatic (binary)
        float(atom.GetTotalValence()) / 6.0,  # Valence (normalized)
        float(atom.GetNumRadicalElectrons()),  # Num radical electrons (numeric)
        float(atom.IsInRing()),  # Is in ring (binary)
    ]

    # 3. Chirality
    if use_chirality:
        # Use a defined list of chiral tags for robust one-hot encoding if needed,
        # but using the integer value (0-3) is simpler for GNNs:
        # 0: CHI_UNSPECIFIED, 1: CHI_TETRAHEDRAL_CW, 2: CHI_TETRAHEDRAL_CCW, 3: CHI_OTHER
        chiral_type = int(atom.GetChiralTag())
        features.append(float(chiral_type))

    return features


def get_bond_features(
    bond: Chem.rdchem.Bond, use_stereochemistry: bool = True
) -> List[float]:
    """
    Convert RDKit bond object to feature vector.
    """
    # 1. Bond type (One-hot encoding)
    bond_type = bond.GetBondType()
    bond_type_enc = [float(bond_type == x) for x in PERMITTED_BOND_TYPES]

    # 2. Basic properties (Binary)
    features = bond_type_enc + [
        float(bond.GetIsConjugated()),  # Is conjugated
        float(bond.IsInRing()),  # Is in ring
    ]

    # 3. Stereochemistry
    if use_stereochemistry:
        # Use the integer value (0-3) for simplicity:
        # 0: STEREONONE, 1: STEREOANY, 2: STEREOZ, 3: STEREOE
        stereo = int(bond.GetStereo())
        features.append(float(stereo))

    return features


# --- Graph Conversion ---


def smiles_to_graph(
    smiles: str,
    use_chirality: bool = True,
    use_stereochemistry: bool = True,
    add_explicit_hydrogens: bool = False,
) -> Data:
    """
    Convert SMILES string to PyTorch Geometric Data object.

    *Debug Note: Ensures explicit error handling and correct empty graph setup.*
    """
    mol = Chem.MolFromSmiles(smiles)

    # **R1: Robustness Check for Invalid SMILES**
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # **CRITICAL FIX:** Extract only the largest fragment (desalting/desolventing)
    fragments = GetMolFrags(mol, asMols=True)
    if fragments:
        # Find the fragment with the most atoms (typically the main drug molecule)
        largest_mol = max(fragments, key=lambda m: m.GetNumAtoms())
        mol = largest_mol

    # Add explicit hydrogens if requested
    if add_explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if add_explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # 1. Get Node Features (x)
    node_features = [
        get_atom_features(atom, use_chirality=use_chirality)
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. Get Edge Indices (edge_index) and Edge Features (edge_attr)
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = get_bond_features(
            bond, use_stereochemistry=use_stereochemistry
        )

        # Add both directions for an undirected graph
        edge_indices.append([i, j])
        edge_features.append(bond_feat)

        edge_indices.append([j, i])
        edge_features.append(bond_feat)

    # **R2: Simplified Edge Handling for Single-Atom Molecules**
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Calculate dimension based on feature creation logic
        bond_feat_dim = (
            len(PERMITTED_BOND_TYPES) + 2 + (1 if use_stereochemistry else 0)
        )
        edge_attr = torch.empty((0, bond_feat_dim), dtype=torch.float)
    else:
        # .t().contiguous() is crucial for PyG's COO format
        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        )
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # 3. Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# --- Validation and Descriptors (Minor Refinement) ---


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is chemically valid.
    """
    # Simplified validation: MolFromSmiles returns None for most invalid SMILES
    return Chem.MolFromSmiles(smiles) is not None


def get_molecular_descriptors(smiles: str) -> dict:
    """
    Calculate common molecular descriptors from SMILES.

    *Debug Note: Uses Descriptors.calcMolDescriptors for a slightly cleaner interface
    but your original usage of individual functions is also fine.*
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Use the specific descriptors you defined (MolWt, MolLogP, etc.)
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
