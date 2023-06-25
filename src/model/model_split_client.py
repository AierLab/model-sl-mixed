import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from comn import ClientSocket
import pickle
from abc import abstractmethod, ABC
from typing import Tuple

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import AbstractModel
from torch.nn import CrossEntropyLoss


class SplitClientModel(AbstractModel):
    def __init__(self, model_layers, socket: ClientSocket, model_dir: str):
        super().__init__(model_dir)
        self.socket = None
        # get all model layers
        self.layers = nn.ModuleList(list(model_layers.children()))
        self.server_data = None
        self.socket = socket
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate all layers
        layer_index = 0
        while layer_index < len(self.layers):

            # If not the first layer, the input is the result from the server
            if self.server_data is not None:
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

            # receive the result from the server
            print("Waiting intermediate result from the server")
            self.server_data = self.socket.receive_data()
        return x

    def backward(self):
        """Compute backward result of all model layers."""
        # iterate all layers in reverse order
        layer_index = len(self.layers) - 1

        grads = self.layers[layer_index].backward()

        # Send the result back to the client
        serialized_data = pickle.dumps(grads)

        print("Sending grads result to the server")
        self.socket.send_data(serialized_data)

        while layer_index >= 0:
            print("Waiting intermediate result from the server")
            serialized_data = self.socket.receive_data()
            grads = pickle.loads(serialized_data)

            grads = self.layers[layer_index].backward(grads)

            # Send the result back to the client
            serialized_data = pickle.dumps(grads)
            print("Sending grads result to the server")
            self.socket.send_data(serialized_data)

            layer_index -= 1

            # # Save the tensor data to a local file # FIXME never used, may need to be removed
            # torch.save(tensor_data, f'../tmp/client/layer_{layer_index}_grads_input.pt')
            # print("Sending intermediate result back to the client")
            # Compute the backward result of the tensor data

    def model_train(self, dataloader: DataLoader, epochs: int, device: torch.device):
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

        for epoch in range(base_epoch, epochs):
            loss = None
            for i, (inputs, labels) in enumerate(dataloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                self.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            self.save_local(epoch, loss, optimizer.state_dict())

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO: Implement this method
        pass
