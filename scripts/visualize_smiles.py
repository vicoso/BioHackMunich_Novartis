#!/usr/bin/env python3
"""
SMILES Molecular Graph Visualizer

This script visualizes SMILES strings as attributed graphs with detailed
node and edge features. It provides interactive visualization with feature
inspection capabilities.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
from matplotlib.widgets import Button
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from rdkit.Chem import Descriptors
except ImportError:
    Descriptors = None
import torch
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_path)

try:
    from data.molecular_features import (
        smiles_to_graph,
        get_atom_features,
        get_bond_features,
        get_molecular_descriptors,
    )
    from utils.constants import PERMITTED_ATOMS, PERMITTED_BOND_TYPES
except ImportError:
    # Fallback if imports fail - define minimal versions
    print(
        "Warning: Could not import from src modules. Using fallback definitions."
    )

    # Minimal fallback constants
    PERMITTED_ATOMS = [
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

    PERMITTED_BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    # Minimal fallback functions
    def get_atom_features_fallback(atom, use_chirality=True):
        """Minimal fallback atom features."""
        atom_type = atom.GetSymbol()
        atom_type_enc = [float(atom_type == x) for x in PERMITTED_ATOMS]

        features = atom_type_enc + [
            float(atom.GetTotalNumHs(includeNeighbors=True)) / 8.0,
            float(atom.GetTotalDegree()) / 4.0,
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalValence()) / 6.0,
            float(atom.GetNumRadicalElectrons()),
            float(atom.IsInRing()),
        ]

        if use_chirality:
            features.append(float(int(atom.GetChiralTag())))

        return features

    def get_bond_features_fallback(bond, use_stereochemistry=True):
        """Minimal fallback bond features."""
        bond_type = bond.GetBondType()
        bond_type_enc = [float(bond_type == x) for x in PERMITTED_BOND_TYPES]

        features = bond_type_enc + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

        if use_stereochemistry:
            features.append(float(int(bond.GetStereo())))

        return features

    def smiles_to_graph_fallback(
        smiles,
        use_chirality=True,
        use_stereochemistry=True,
        add_explicit_hydrogens=False,
    ):
        """Minimal fallback graph conversion."""
        from torch_geometric.data import Data

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        if add_explicit_hydrogens:
            mol = Chem.AddHs(mol)

        # Node features
        node_features = [
            get_atom_features_fallback(atom, use_chirality)
            for atom in mol.GetAtoms()
        ]
        x = torch.tensor(node_features, dtype=torch.float)

        # Edge features
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = get_bond_features_fallback(bond, use_stereochemistry)

            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(
                (
                    0,
                    len(PERMITTED_BOND_TYPES)
                    + 2
                    + (1 if use_stereochemistry else 0),
                ),
                dtype=torch.float,
            )
        else:
            edge_index = (
                torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            )
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_molecular_descriptors_fallback(smiles):
        """Minimal fallback molecular descriptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Calculate basic properties
        result = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "molecular_weight": 0.0,
            "logp": 0.0,
            "num_rings": 0,
            "tpsa": 0.0,
        }

        # Try to get more descriptors if available
        if Descriptors is not None:
            try:
                if hasattr(Descriptors, "MolWt"):
                    result["molecular_weight"] = Descriptors.MolWt(mol)
                if hasattr(Descriptors, "MolLogP"):
                    result["logp"] = Descriptors.MolLogP(mol)
                if hasattr(Descriptors, "RingCount"):
                    result["num_rings"] = Descriptors.RingCount(mol)
                if hasattr(Descriptors, "TPSA"):
                    result["tpsa"] = Descriptors.TPSA(mol)
            except Exception:
                pass  # Keep default values

        return result

    # Override functions if imports failed
    if "smiles_to_graph" not in globals():
        smiles_to_graph = smiles_to_graph_fallback
        get_atom_features = get_atom_features_fallback
        get_bond_features = get_bond_features_fallback
        get_molecular_descriptors = get_molecular_descriptors_fallback


class SMILESGraphVisualizer:
    """Interactive SMILES graph visualizer with feature inspection."""

    def __init__(self, smiles: str, show_features: bool = True):
        """
        Initialize the visualizer.

        Args:
            smiles: SMILES string to visualize
            show_features: Whether to display feature values in the visualization
        """
        self.smiles = smiles
        self.show_features = show_features
        self.mol = Chem.MolFromSmiles(smiles)

        if self.mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Convert to PyG graph
        self.graph_data = smiles_to_graph(smiles)

        # Create NetworkX graph for visualization
        self.nx_graph = self._create_networkx_graph()

        # Feature information for display
        self.atom_feature_names = self._get_atom_feature_names()
        self.bond_feature_names = self._get_bond_feature_names()

        # Visualization state
        self.selected_node = None
        self.selected_edge = None

    def _get_atom_feature_names(self) -> List[str]:
        """Get names for atom features."""
        names = []

        # Atom types (one-hot)
        names.extend([f"is_{atom}" for atom in PERMITTED_ATOMS])

        # Basic properties
        names.extend(
            [
                "num_hydrogens_norm",
                "degree_norm",
                "formal_charge",
                "is_aromatic",
                "valence_norm",
                "num_radical_electrons",
                "is_in_ring",
                "chirality_tag",
            ]
        )

        return names

    def _get_bond_feature_names(self) -> List[str]:
        """Get names for bond features."""
        names = []

        # Bond types (one-hot)
        bond_type_names = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        names.extend([f"is_{bond_type}" for bond_type in bond_type_names])

        # Basic properties
        names.extend(["is_conjugated", "is_in_ring", "stereo_tag"])

        return names

    def _create_networkx_graph(self) -> nx.Graph:
        """Convert PyG graph to NetworkX for visualization."""
        G = nx.Graph()

        # Add nodes with atom information
        for i, atom in enumerate(self.mol.GetAtoms()):
            G.add_node(
                i,
                symbol=atom.GetSymbol(),
                atomic_num=atom.GetAtomicNum(),
                features=self.graph_data.x[i].numpy(),
            )

        # Add edges with bond information
        edge_index = self.graph_data.edge_index.numpy()
        edge_attr = self.graph_data.edge_attr.numpy()

        # Only add unique edges (undirected graph)
        added_edges = set()
        edge_idx = 0

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            edge_tuple = tuple(sorted([u, v]))

            if edge_tuple not in added_edges:
                added_edges.add(edge_tuple)

                # Get bond from molecule
                bond = self.mol.GetBondBetweenAtoms(int(u), int(v))
                if bond:
                    G.add_edge(
                        u,
                        v,
                        bond_type=str(bond.GetBondType()),
                        features=edge_attr[i],
                    )

        return G

    def get_node_colors(self) -> List[str]:
        """Get colors for nodes based on atom types."""
        color_map = {
            "C": "#909090",  # Carbon - gray
            "N": "#3050F8",  # Nitrogen - blue
            "O": "#FF0D0D",  # Oxygen - red
            "S": "#FFFF30",  # Sulfur - yellow
            "P": "#FF8000",  # Phosphorus - orange
            "F": "#90E050",  # Fluorine - light green
            "Cl": "#1FF01F",  # Chlorine - green
            "Br": "#A62929",  # Bromine - brown
            "I": "#940094",  # Iodine - purple
            "H": "#FFFFFF",  # Hydrogen - white
        }

        colors = []
        for node in self.nx_graph.nodes():
            symbol = self.nx_graph.nodes[node]["symbol"]
            colors.append(color_map.get(symbol, "#FFC0CB"))  # Default pink

        return colors

    def get_edge_colors_and_widths(self) -> Tuple[List[str], List[float]]:
        """Get colors and widths for edges based on bond types."""
        color_map = {
            "SINGLE": "#000000",  # Black
            "DOUBLE": "#FF0000",  # Red
            "TRIPLE": "#0000FF",  # Blue
            "AROMATIC": "#00FF00",  # Green
        }

        width_map = {
            "SINGLE": 2.0,
            "DOUBLE": 3.0,
            "TRIPLE": 4.0,
            "AROMATIC": 2.5,
        }

        colors = []
        widths = []

        for edge in self.nx_graph.edges():
            bond_type = self.nx_graph.edges[edge]["bond_type"]
            colors.append(color_map.get(bond_type, "#000000"))
            widths.append(width_map.get(bond_type, 2.0))

        return colors, widths

    def format_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.01,
    ) -> str:
        """Format features for display, showing only significant values."""
        lines = []

        for name, value in zip(feature_names, features):
            if (
                abs(value) > threshold or value == 1.0
            ):  # Show significant or binary features
                lines.append(f"{name}: {value:.3f}")

        return "\n".join(
            lines[:10]
        )  # Limit to first 10 features for readability

    def on_click(self, event):
        """Handle click events for both nodes and edges."""
        if event.inaxes != self.ax:
            return

        click_x, click_y = event.xdata, event.ydata

        # Check for node clicks first
        min_node_dist = float("inf")
        closest_node = None

        for node, (x, y) in self.pos.items():
            dist = ((click_x - x) ** 2 + (click_y - y) ** 2) ** 0.5
            if dist < min_node_dist:
                min_node_dist = dist
                closest_node = node

        # Check for edge clicks
        min_edge_dist = float("inf")
        closest_edge = None

        for edge in self.nx_graph.edges():
            u, v = edge
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]

            # Calculate distance from click to edge (line segment)
            edge_dist = self._point_to_line_distance(
                click_x, click_y, x1, y1, x2, y2
            )

            if edge_dist < min_edge_dist:
                min_edge_dist = edge_dist
                closest_edge = edge

        # Determine what was clicked (prioritize nodes if both are close)
        node_threshold = 0.1
        edge_threshold = 0.05

        if min_node_dist < node_threshold:
            # Node click
            self.selected_node = closest_node
            self.selected_edge = None
            self.update_info_panel()
        elif min_edge_dist < edge_threshold:
            # Edge click
            self.selected_node = None
            self.selected_edge = closest_edge
            self.update_info_panel()

    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment."""
        # Vector from point 1 to point 2
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            # Points are the same
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Parameter t for closest point on line
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # Clamp t to [0, 1] to stay on line segment
        t = max(0, min(1, t))

        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance from click point to closest point on line
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def update_info_panel(self):
        """Update the information panel with selected node/edge details."""
        info_text = f"SMILES: {self.smiles}\n\n"

        if self.selected_node is not None:
            node_data = self.nx_graph.nodes[self.selected_node]
            info_text += f"Node {self.selected_node}:\n"
            info_text += (
                f"Atom: {node_data['symbol']} (Z={node_data['atomic_num']})\n\n"
            )

            if self.show_features:
                info_text += "Node Features:\n"
                features_str = self.format_features(
                    node_data["features"], self.atom_feature_names
                )
                info_text += features_str

        elif self.selected_edge is not None:
            edge_data = self.nx_graph.edges[self.selected_edge]
            u, v = self.selected_edge
            atom_u = self.nx_graph.nodes[u]["symbol"]
            atom_v = self.nx_graph.nodes[v]["symbol"]

            info_text += f"Edge {u}-{v}:\n"
            info_text += f"Bond: {atom_u}-{atom_v}\n"
            info_text += f"Type: {edge_data['bond_type']}\n\n"

            if self.show_features and "features" in edge_data:
                info_text += "Edge Features:\n"
                features_str = self.format_features(
                    edge_data["features"], self.bond_feature_names
                )
                info_text += features_str

        self.info_text.set_text(info_text)
        self.fig.canvas.draw()

    def visualize(
        self,
        figsize: Tuple[int, int] = (16, 10),
        node_size: int = 500,
        font_size: int = 10,
    ) -> Figure:
        """
        Create interactive visualization of the molecular graph.

        Args:
            figsize: Figure size (width, height)
            node_size: Size of nodes in the graph
            font_size: Font size for labels

        Returns:
            matplotlib Figure object
        """
        self.fig, (self.ax, self.info_ax) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )

        # Calculate layout
        self.pos = nx.spring_layout(self.nx_graph, seed=42, k=1, iterations=50)

        # Get colors and styles
        node_colors = self.get_node_colors()
        edge_colors, edge_widths = self.get_edge_colors_and_widths()

        # Draw the graph
        nx.draw_networkx_edges(
            self.nx_graph,
            self.pos,
            ax=self.ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
        )

        nx.draw_networkx_nodes(
            self.nx_graph,
            self.pos,
            ax=self.ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=1,
        )

        # Add atom labels
        labels = {
            node: self.nx_graph.nodes[node]["symbol"]
            for node in self.nx_graph.nodes()
        }
        nx.draw_networkx_labels(
            self.nx_graph,
            self.pos,
            labels,
            ax=self.ax,
            font_size=font_size,
            font_weight="bold",
        )

        self.ax.set_title(
            f"Molecular Graph: {self.smiles}", fontsize=14, fontweight="bold"
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Information panel
        self.info_ax.axis("off")
        self.info_text = self.info_ax.text(
            0.05,
            0.95,
            "",
            transform=self.info_ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Add legend
        self._add_legend()

        # Connect click event
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Initialize info panel
        self.update_info_panel()

        return self.fig

    def _add_legend(self):
        """Add legend for atom and bond types."""
        # Atom type legend
        atom_legend_elements = []
        common_atoms = ["C", "N", "O", "S", "P", "F", "Cl", "Br"]
        color_map = {
            "C": "#909090",
            "N": "#3050F8",
            "O": "#FF0D0D",
            "S": "#FFFF30",
            "P": "#FF8000",
            "F": "#90E050",
            "Cl": "#1FF01F",
            "Br": "#A62929",
        }

        legend_text = "Atom Colors:\n"
        for atom in common_atoms:
            if any(
                self.nx_graph.nodes[n]["symbol"] == atom
                for n in self.nx_graph.nodes()
            ):
                legend_text += f"● {atom}\n"

        legend_text += "\nBond Types:\n"
        legend_text += "— Single\n"
        legend_text += "— Double (red, thick)\n"
        legend_text += "— Triple (blue, thick)\n"
        legend_text += "— Aromatic (green)\n"

        legend_text += "\nClick on atoms or bonds to see features!"

        self.info_ax.text(
            0.05,
            0.05,
            legend_text,
            transform=self.info_ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
        )

    def show_rdkit_structure(self) -> None:
        """Display RDKit 2D structure alongside the graph."""
        img = Draw.MolToImage(self.mol, size=(300, 300))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(f"RDKit Structure: {self.smiles}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def print_graph_statistics(self) -> None:
        """Print detailed statistics about the molecular graph."""
        print(f"\n{'='*60}")
        print(f"MOLECULAR GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"SMILES: {self.smiles}")
        print(f"Number of atoms: {self.mol.GetNumAtoms()}")
        print(f"Number of bonds: {self.mol.GetNumBonds()}")
        print(f"Graph nodes: {self.nx_graph.number_of_nodes()}")
        print(f"Graph edges: {self.nx_graph.number_of_edges()}")

        # Atom type distribution
        atom_counts = {}
        for node in self.nx_graph.nodes():
            symbol = self.nx_graph.nodes[node]["symbol"]
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

        print(f"\nAtom distribution:")
        for atom, count in sorted(atom_counts.items()):
            print(f"  {atom}: {count}")

        # Bond type distribution
        bond_counts = {}
        for edge in self.nx_graph.edges():
            bond_type = self.nx_graph.edges[edge]["bond_type"]
            bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1

        print(f"\nBond distribution:")
        for bond_type, count in sorted(bond_counts.items()):
            print(f"  {bond_type}: {count}")

        # Feature dimensions
        print(f"\nFeature dimensions:")
        print(f"  Node features: {self.graph_data.x.shape}")
        print(f"  Edge features: {self.graph_data.edge_attr.shape}")

        # Molecular descriptors
        descriptors = get_molecular_descriptors(self.smiles)

        print(f"\nMolecular descriptors:")
        for name, value in descriptors.items():
            print(f"  {name}: {value:.3f}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize SMILES strings as attributed graphs"
    )
    parser.add_argument(
        "smiles",
        nargs="?",
        default="CCO",  # Ethanol as default
        help="SMILES string to visualize (default: CCO)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Do not show feature values in visualization",
    )
    parser.add_argument(
        "--show-rdkit", action="store_true", help="Also show RDKit 2D structure"
    )
    parser.add_argument(
        "--node-size",
        type=int,
        default=500,
        help="Size of nodes in visualization (default: 500)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=10,
        help="Font size for labels (default: 10)",
    )
    parser.add_argument(
        "--save", type=str, help="Save visualization to file (e.g., output.png)"
    )
    parser.add_argument(
        "--examples", action="store_true", help="Show example SMILES strings"
    )

    args = parser.parse_args()

    if args.examples:
        print("Example SMILES strings:")
        examples = [
            ("CCO", "Ethanol"),
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
            ("CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen"),
            ("C1=CC=C(C=C1)CCN", "Phenethylamine"),
            ("c1ccc2c(c1)oc1ccccc12", "Dibenzofuran"),
        ]
        for smiles, name in examples:
            print(f"  {smiles:<35} # {name}")
        return

    try:
        # Create visualizer
        visualizer = SMILESGraphVisualizer(
            args.smiles, show_features=not args.no_features
        )

        # Print statistics
        visualizer.print_graph_statistics()

        # Create visualization
        fig = visualizer.visualize(
            node_size=args.node_size, font_size=args.font_size
        )

        # Save if requested
        if args.save:
            fig.savefig(args.save, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {args.save}")

        # Show RDKit structure if requested
        if args.show_rdkit:
            visualizer.show_rdkit_structure()

        # Show interactive plot
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
