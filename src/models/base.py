"""
Base model classes and interfaces.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all molecular property prediction models.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data: Input data (typically PyTorch Geometric Data/Batch)

        Returns:
            Model predictions
        """
        pass

    def save_model(self, filepath: str, save_config: bool = True) -> None:
        """
        Save model state dict and optionally configuration.

        Args:
            filepath: Path to save the model
            save_config: Whether to save model configuration
        """
        save_dict = {
            "state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
        }

        if save_config:
            save_dict["config"] = self.config

        torch.save(save_dict, filepath)

    @classmethod
    def load_model(cls, filepath: str, map_location: Optional[str] = None):
        """
        Load model from saved state dict.

        Args:
            filepath: Path to the saved model
            map_location: Device to load the model on

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=map_location)

        if "config" not in checkpoint:
            raise ValueError("Model configuration not found in checkpoint")

        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])

        return model

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_info(self) -> Dict[str, int]:
        """Get detailed parameter information by layer."""
        param_info = {}
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            param_count = param.numel()
            param_info[name] = param_count
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count

        param_info["total_parameters"] = total_params
        param_info["trainable_parameters"] = trainable_params
        param_info["non_trainable_parameters"] = total_params - trainable_params

        return param_info
