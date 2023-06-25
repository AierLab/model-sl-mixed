import os
from typing import Tuple

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class AbstractModel(nn.Module, ABC):

    def __init__(self, model_dir: str) -> None:
        """
        client: client instance
        model_dir: the folder for model state dict, pt file
        """
        super(AbstractModel, self).__init__()
        self.model_dir = model_dir

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def model_train(self, dataloader: DataLoader, epochs: int, device: torch.device = None):
        # TODO better to have loss and optimizer defied outside, and pass as parameters.
        pass

    @abstractmethod
    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        pass

    def save_local(self, epoch: int, loss, optimizer_state_dict: dict) -> None:
        """
        Save to local.
        """
        path = os.path.join(self.model_dir, f"{type(self).__name__}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
        }, path)

    def load_local(self) -> dict:
        """
        Save to local.
        """
        path = os.path.join(self.model_dir, f"{type(self).__name__}.pt")
        if os.path.exists(path):
            print("Log: Loaded model state dict.")
            return torch.load(path)
