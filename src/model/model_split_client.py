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
from splitlearn.client import SplitClient

class SplitClientModel(AbstractModel):

    def __init__(self, model_dir: str, model_layers) -> None:
        super(SplitClientModel, self).__init__(model_dir)
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
    
    def forward_all(self, input: torch.Tensor) -> torch.Tensor:

        """Compute forward result of all model layers."""

        # iterate all layers
        layer_index = 0
        server_data = None
        while layer_index < len(self.layers):
            
            # If not the first layer, the input is the result from the server
            if server_data is not None:
                input = pickle.loads(server_data)
                input.to(self.device)
                # Save the input tensor to a local file
                torch.save(input, '../tmp/client/layer_{layer_index}_input.pt')

            # Compute the forward result of the tensor data
            data = self.forward(input, layer_index)
            # pickle the tensor data
            serialized_data = pickle.dumps(data) 
            # Save the tensor data to a local file
            torch.save(serialized_data, '../tmp/client/layer_{layer_index}_output.pt')

            # Send the result to the server
            # TODO Don't need to send the data to the server if it is the last layer
            print("Sending intermediate result to the server")
            self.socket.send(serialized_data)
            layer_index += 1   

            # receive the result from the server
            server_data = self.recv_data()
        return data


    def backward_all(self, loss):
        
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

            # Save the tensor data to a local file
            torch.save(tensor_data, '../tmp/client/layer_{layer_index}_grads_input.pt')
            print("Sending intermediate result back to the client")
            # Compute the backward result of the tensor data
            result = self.backward(tensor_data, layer_index)
            serialized_data = pickle.dumps(result)
            # Send the result back to the client
            client_socket.send(serialized_data)
            layer_index -= 1   
    
    def recv_data(self):
        data = b""
        while True:
            packet = self.socket.recv(1024)
            if not packet: 
                print("No more data from client")
                break
            data += packet
        return data

    def model_train(self, epochs: int):
        """Train the network on the training set."""
        self.to(self.device)
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

        
        for epoch in range(base_epoch, epochs):
            for i, data in enumerate(self.client.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.forward_all(inputs)
                loss = criterion(outputs, labels)
                # TODO: backward_all
                self.backward_all(loss)
                optimizer.step()
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            self.save_local(epoch, loss, optimizer.state_dict())
            

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        