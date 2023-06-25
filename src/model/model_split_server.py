import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from typing import Tuple
import pickle
import socket

from .model_abstract import AbstractModel
from comn import AbstractClient
from comn import AbstractServer


class SplitServerModel(AbstractModel):

    def __init__(self, model_dir: str, model_layers) -> None:
        super(SplitServerModel, self).__init__(model_dir)
        self.socket = None
        self.model_dir = model_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get all model layers in server-side
        self.layers = []
        for layer in model_layers.children():
            self.layers.append(layer)

    def set_socket(self, socket: socket):
        self.socket = socket

    def forward(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Compute forward result of a specific model layer."""
        return self.layers[layer_index].forward(x)

    def forward_all(self):

        """Compute forward result of all model layers."""

        # Wait for a connection
        print("Waiting for a connection...")
        client_socket, client_address = self.socket.accept()
        print(f"Connection from {client_address}")

        # iterate all layers
        layer_index = 0
        while layer_index < len(self.layers):
            data = self.recv_data(client_socket)
            tensor_data = pickle.loads(data)
            # Save the input tensor to a local file
            torch.save(tensor_data, '../tmp/server/layer_{layer_index}_input.pt')

            # Compute the forward result of the tensor data
            result = self.forward(tensor_data, layer_index)
            serialized_data = pickle.dumps(result)
            # Save the output tensor to a local file 
            torch.save(serialized_data, '../tmp/server/layer_{layer_index}_output.pt')
            # Send the result back to the client
            print("Sending intermediate result back to the client")
            client_socket.send(serialized_data)
            layer_index += 1

    def backward_all(self):

        """Compute backward result of all model layers."""

        # Wait for a connection
        print("Waiting for a connection...")
        client_socket, client_address = self.socket.accept()
        print(f"Connection from {client_address}")

        # iterate all layers in reverse order
        layer_index = len(self.layers) - 1
        while layer_index >= 0:
            grads = self.recv_data(client_socket)
            tensor_data = pickle.loads(grads)
            # Save the received grads to a local file
            torch.save(tensor_data, '../tmp/server/layer_{layer_index}_grads_input.pt')

            # Compute the backward result of the tensor data
            # TODO: override backward function
            result = self.backward(tensor_data, layer_index)
            serialized_data = pickle.dumps(result)

            # Save the output grads to a local file
            torch.save(serialized_data, '../tmp/server/layer_{layer_index}_grads_output.pt')

            # Send the grads back to the client
            print("Sending intermediate result back to the client")
            client_socket.send(serialized_data)
            layer_index -= 1

    def recv_data(self, client_socket):
        data = b""
        while True:
            packet = client_socket.recv(1024)
            if not packet:
                print("No more data from client")
                break
            data += packet
        return data

    def model_train(self, epochs: int, device: torch.device = None):
        """Train the network on the training set."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        base_epoch = 0
        # base_loss = None # TODO not used

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])
            optimizer.load_state_dict(model_dict["optimizer_state_dict"])
            base_epoch = model_dict["epoch"]
            # base_loss = model_dict['loss'] # TODO not used

        #     self.save_local(epoch, loss, optimizer.state_dict())

        self.socket.listen(5)

        self.forward_all()
        self.backward_all()
        optimizer.step()
        optimizer.zero_grad()

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO
