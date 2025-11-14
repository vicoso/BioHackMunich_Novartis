"""
BioHackMunich Novartis: GNN-based Gene Expression Prediction

A modular package for predicting gene expression from molecular SMILES using Graph Neural Networks.
"""

__version__ = "0.1.0"
__author__ = "BioHackMunich Team"

from . import data, models, training, utils, config

__all__ = ["data", "models", "training", "utils", "config"]
