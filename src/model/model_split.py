import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from typing import Tuple
import pickle

from .model_abstract import AbstractModel
from comn import AbstractClient
from comn import AbstractServer
from splitlearn.server import SplitServer

class SplitServerModel(AbstractModel):

    def __init__(self, server: SplitServer, model_dir: str, model) -> None:
        super(SplitServerModel, self).__init__(server, model_dir)
        self.model = model
        self.server = server
        self.model_dir = model_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get all model layers in server-side
        self.layers = []
        for layer in self.model.children():
            self.layers.append(layer)


    def forward(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Compute forward result of a specific model layer."""
        return self.layers[layer_index].forward(x)


    def model_train(self, dataloader: DataLoader, epochs: int, device: torch.device = None):
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

        # for epoch in range(1, epochs+1):
        #     epoch += base_epoch
        #     print(f"Log: start Epoch{epoch}")

        #     loss = None
        #     for images, labels in dataloader:
        #         images, labels = images.to(device), labels.to(device)
        #         optimizer.zero_grad()
        #         loss = criterion(self.forward(images), labels) # TODO double check
        #         loss.backward()
        #         optimizer.step()

        #     self.save_local(epoch, loss, optimizer.state_dict())

        socket = self.server.get_socket
        socket.listen(1)

        while True:
            # Wait for a connection
            print("Waiting for a connection...")
            client_socket, client_address = socket.accept()
            try:
                print(f"Connection from {client_address}")

                layer_index = 0

                data = b""
                while True:
                    packet = client_socket.recv(1024)
                    if not packet: 
                        print("No more data from client")
                        break
                    data += packet

                tensor_data = pickle.loads(data)
                # Save the tensor data to a local file
                torch.save(tensor_data, 'layer_{layer_index}_result.pt')
                print("Sending intermediate result back to the client")
                result = self.forward(tensor_data, layer_index)
                serialized_data = pickle.dumps(result)
                layer_index += 1
                
                client_socket.send(serialized_data)
                    

                    
            finally:
                # Clean up the connection
                client_socket.close()
                print("Connection closed")

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])

        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.forward(images) # TODO double check
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy