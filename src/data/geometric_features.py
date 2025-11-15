import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj
from typing import List
from scott import KILT


def add_forman_ricci_curvature_features(data_list: List[Data]) -> List[Data]:
    """
    Add Forman-Ricci curvature as edge features to a list of PyTorch Geometric data objects.

    Args:
        data_list: List of PyTorch Geometric Data objects from create_dataset function

    Returns:
        List of PyTorch Geometric Data objects with curvature features added to edge_attr
    """
    enhanced_data_list = []

    for data in data_list:
        try:
            # Convert PyG data to NetworkX graph
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)

            # Handle isolated nodes (single atom molecules)
            if G.number_of_edges() == 0:
                # For molecules with no edges, keep original edge_attr (empty tensor)
                enhanced_data_list.append(data)
                continue

            # Compute Forman-Ricci curvature using SCOTT/KILT
            backend = KILT()
            backend.fit(G)
            curvatures = backend.curvature

            # Create edge curvature features
            edge_curvatures = []
            edge_index = data.edge_index.t().cpu().numpy()

            # Handle different output formats from KILT
            if isinstance(curvatures, dict):
                # Dictionary format: edge -> curvature
                for i in range(edge_index.shape[0]):
                    edge = tuple(sorted([edge_index[i, 0], edge_index[i, 1]]))
                    curvature_value = curvatures.get(edge, 0.0)
                    edge_curvatures.append([curvature_value])
            elif isinstance(curvatures, np.ndarray):
                # Array format: assume same order as G.edges()
                edge_list = list(G.edges())
                edge_to_curvature = {
                    tuple(sorted(edge)): curvatures[i]
                    for i, edge in enumerate(edge_list)
                }

                for i in range(edge_index.shape[0]):
                    edge = tuple(sorted([edge_index[i, 0], edge_index[i, 1]]))
                    curvature_value = edge_to_curvature.get(edge, 0.0)
                    edge_curvatures.append([curvature_value])
            else:
                # Fallback: set all curvatures to 0
                print(
                    f"Warning: Unknown curvature format {type(curvatures)}, using zeros"
                )
                for i in range(edge_index.shape[0]):
                    edge_curvatures.append([0.0])

            # Convert to tensor
            curvature_tensor = torch.tensor(edge_curvatures, dtype=torch.float)

            # Concatenate with existing edge features
            if data.edge_attr is not None:
                new_edge_attr = torch.cat(
                    [data.edge_attr, curvature_tensor], dim=1
                )
            else:
                new_edge_attr = curvature_tensor

            # Create new data object with enhanced edge features
            enhanced_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=new_edge_attr,
                y=data.y if hasattr(data, "y") else None,
                mol_features=(
                    data.mol_features if hasattr(data, "mol_features") else None
                ),
            )

            enhanced_data_list.append(enhanced_data)

        except Exception as e:
            print(f"Error computing curvature for molecule: {e}")
            # Keep original data if curvature computation fails
            enhanced_data_list.append(data)
            continue

    return enhanced_data_list


def compute_single_molecule_curvature(data: Data) -> torch.Tensor:
    """
    Compute Forman-Ricci curvature for a single PyTorch Geometric data object.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Tensor of curvature values for each edge
    """
    # Convert PyG data to NetworkX graph
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)

    # Handle case with no edges
    if G.number_of_edges() == 0 or data.edge_index is None:
        num_edges = 0 if data.edge_index is None else data.edge_index.size(1)
        return torch.empty((num_edges, 1), dtype=torch.float)

    # Compute Forman-Ricci curvature
    backend = KILT()
    backend.fit(G)
    curvatures = backend.curvature

    # Map curvatures to edges
    edge_curvatures = []
    if data.edge_index is None:
        return torch.empty((0, 1), dtype=torch.float)
    edge_index = data.edge_index.t().cpu().numpy()

    # Handle different output formats from KILT
    if isinstance(curvatures, np.ndarray):
        # Array format: assume same order as G.edges()
        edge_list = list(G.edges())
        edge_to_curvature = {
            tuple(sorted(edge)): curvatures[i]
            for i, edge in enumerate(edge_list)
        }

        for i in range(edge_index.shape[0]):
            edge = tuple(sorted([edge_index[i, 0], edge_index[i, 1]]))
            curvature_value = edge_to_curvature.get(edge, 0.0)
            edge_curvatures.append([curvature_value])
    else:
        # Fallback: set all curvatures to 0
        for i in range(edge_index.shape[0]):
            edge_curvatures.append([0.0])

    return torch.tensor(edge_curvatures, dtype=torch.float)
