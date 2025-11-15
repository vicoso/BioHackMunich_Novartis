#!/usr/bin/env python3
"""
Demo script for SMILES molecular graph visualization.

This script demonstrates various molecules and their graph representations.
"""

import os
import sys

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    from visualize_smiles import SMILESGraphVisualizer
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure matplotlib, networkx, torch, and rdkit are installed.")
    sys.exit(1)


def demo_molecules():
    """Demonstrate visualization of various molecules."""

    molecules = [
        ("CCO", "Ethanol"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen"),
    ]

    print("=== SMILES Molecular Graph Visualization Demo ===\n")

    for smiles, name in molecules:
        print(f"Analyzing {name} ({smiles})...")

        try:
            # Create visualizer
            visualizer = SMILESGraphVisualizer(smiles, show_features=True)

            # Print basic statistics
            print(f"  Atoms: {visualizer.nx_graph.number_of_nodes()}")
            print(f"  Bonds: {visualizer.nx_graph.number_of_edges()}")

            # Get atom distribution
            atom_counts = {}
            for node in visualizer.nx_graph.nodes():
                symbol = visualizer.nx_graph.nodes[node]["symbol"]
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

            atom_str = ", ".join(
                [
                    f"{atom}:{count}"
                    for atom, count in sorted(atom_counts.items())
                ]
            )
            print(f"  Composition: {atom_str}")

            # Save visualization
            fig = visualizer.visualize(figsize=(12, 8))
            filename = f"{name.lower().replace(' ', '_')}_graph.png"
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → Saved visualization to {filename}")

        except Exception as e:
            print(f"  Error processing {name}: {e}")

        print()

    print("Demo completed! Check the generated PNG files for visualizations.")


def interactive_demo():
    """Interactive demo where user can input SMILES."""

    print("=== Interactive SMILES Visualization ===")
    print("Enter SMILES strings to visualize (or 'quit' to exit)")
    print("Examples: CCO, CC(=O)O, c1ccccc1")

    while True:
        smiles = input("\nEnter SMILES: ").strip()

        if smiles.lower() in ["quit", "exit", "q"]:
            break

        if not smiles:
            continue

        try:
            visualizer = SMILESGraphVisualizer(smiles, show_features=True)
            visualizer.print_graph_statistics()

            # Show visualization
            fig = visualizer.visualize()
            plt.show()

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo SMILES molecular graph visualization"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run interactive demo"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        demo_molecules()
