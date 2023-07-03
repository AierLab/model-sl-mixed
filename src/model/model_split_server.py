import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

import pickle
from abc import abstractmethod, ABC
from typing import Tuple, List

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from helper import NoneException
from model import AbstractModel


class SplitServerModel(AbstractModel):
    def __init__(self, model_layers, model_dir: str, device=None):
        super().__init__(model_dir)
        # get all model layers
        self.layers = nn.ModuleList(list(model_layers.children()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.forward_results: List[torch.Tensor] = []
        self.layer_index = 0

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate one layers
        x = x.to(self.device)

        # Compute the forward result of the tensor data
        self.forward_results.append(x)
        x = self.layers[self.layer_index](x)
        self.forward_results.append(x)

        self.layer_index += 1

        return x

    def backward(self, grad: torch.Tensor):
        """Compute backward result of all model layers."""
        # iterate all layers in reverse order

        last_forward_result = self.forward_results.pop()
        last_forward_result.grad = grad

        forward_result = self.forward_results.pop()
        forward_result.grad = torch.autograd.grad(outputs=last_forward_result,
                                                  inputs=forward_result,
                                                  grad_outputs=torch.ones_like(last_forward_result),
                                                  allow_unused=True)[0]

        self.layer_index -= 1

        return forward_result.grad

    def data_process(self, data: dict):
        """
        Train the network on the training set.
        """

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])

        if data["stage"] == "forward":
            serialized_data = data["byte_data"]
            x = pickle.loads(serialized_data)
            x = self.forward(x)
            data["byte_data"] = pickle.dumps(x)
            return data
        else: # stage == "backward"
            serialized_data = data["byte_data"]
            grad = pickle.loads(serialized_data)
            grad = self.backward(grad)
            data["byte_data"] = pickle.dumps(grad)
            return data

        # while True:
        #     try:
        #         self.forward()
        #         self.backward()
        #         # print("________________________________________________")
        #     except NoneException as e:
        #         print(e)
        #         print("Passive repeat")
        #     except Exception as e:
        #         print(e)
        #         self.socket.send_data(b"")
        #         print("Active repeat")

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO: Implement this method
        pass
