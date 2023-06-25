import pickle
from abc import abstractmethod, ABC
from typing import Tuple

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from comn import AbstractSocket
from model import AbstractModel


class AbstractSplitModel(AbstractModel, ABC):
    def __init__(self, model_layers, socket: AbstractSocket, model_dir: str):
        super().__init__(model_dir)
        self.socket = None
        # get all model layers
        self.layers = nn.ModuleList(list(model_layers.children()))
        self.server_data = None
        self.socket = socket
