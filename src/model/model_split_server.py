import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from comn import AbstractSocket
import pickle
from abc import abstractmethod, ABC
from typing import Tuple

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import AbstractModel


class SplitServerModel(AbstractModel):
    def __init__(self, model_layers, socket: AbstractSocket, model_dir: str):
        super().__init__(model_dir)
        self.socket = None
        # get all model layers
        self.layers = nn.ModuleList(list(model_layers.children()))
        self.server_data = None
        self.socket = socket

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate all layers
        layer_index = 0
        while layer_index < len(self.layers):
            # receive the result from the server
            self.server_data = self.socket.receive_data()  # Server receive data first
            x = pickle.loads(self.server_data)
            x = x.to(self.device)

            # # Save the input tensor to a local file # FIXME never used, may need to be removed
            # torch.save(x, f'../tmp/client/{type(self).__name__}/layer_{layer_index}_input.pt')

            # Compute the forward result of the tensor data
            x = self.layers[layer_index](x)

            # pickle the tensor data
            serialized_data = pickle.dumps(x)
            # # Save the tensor data to a local file # FIXME never used, may need to be removed
            # torch.save(serialized_data, f'../tmp/client/{type(self).__name__}/layer_{layer_index}_output.pt')

            # Send the result to the server
            # TODO Don't need to send the data to the server if it is the last layer
            print("Sending intermediate result to the server")
            self.socket.send_data(serialized_data)
            layer_index += 1
        return x

    def backward(self):
        """Compute backward result of all model layers."""
        # iterate all layers in reverse order
        layer_index = len(self.layers) - 1

        while layer_index >= 0:
            serialized_data = self.socket.receive_data()
            grads = pickle.loads(serialized_data)

            grads = self.layers[layer_index].backward(grads)

            # Send the result back to the client
            serialized_data = pickle.dumps(grads)
            self.socket.send_data(serialized_data)

            layer_index -= 1

            # # Save the tensor data to a local file # FIXME never used, may need to be removed
            # torch.save(tensor_data, f'../tmp/client/layer_{layer_index}_grads_input.pt')
            # print("Sending intermediate result back to the client")
            # Compute the backward result of the tensor data

    def model_train(self, dataloader: DataLoader = None, epochs: int = None, device: torch.device = None):
        """
        Train the network on the training set.
        """
        self.to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=0.001)
        base_epoch = 0
        # base_loss = None # TODO not used

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])
            optimizer.load_state_dict(model_dict["optimizer_state_dict"])
            base_epoch = model_dict["epoch"]
            # base_loss = model_dict['loss'] # TODO not used

        while True:
            optimizer.zero_grad()
            self.forward()
            self.backward()

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO: Implement this method
        pass
