
import socket
import os
import torch

import helper
from model import SplitServerModel


class SplitServer:
    def __init__(self, model: SplitServerModel):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run(self):
        self.model.model_train()
