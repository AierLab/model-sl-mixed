import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

from model.chatglm_6b_split_server.modeling_chatglm import InvalidScoreLogitsProcessor
from splitlearn import SplitClient
import pickle
from abc import abstractmethod, ABC
from typing import Tuple, List, Optional

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from helper import NoneException
from model import AbstractModel
from torch.nn import CrossEntropyLoss


class SplitClientModel(AbstractModel):
    def __init__(self, model_layers, client: SplitClient, model_dir: str, device=None):
        super().__init__(model_dir)
        # get all model layers
        self.layers = model_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # self.optimizers = [Adam(layer.parameters(), lr=0.001, ) for layer in self.layers]
        self.client = client
        self.forward_results: List[torch.Tensor] = []
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate all layers
        layer_index = 0
        self.forward_results = []
        while layer_index < len(self.layers):
            # Compute the forward result of the tensor data
            self.forward_results.append(x)
            x = self.layers[layer_index](x)
            self.forward_results.append(x)

            # pickle the tensor data
            serialized_data = pickle.dumps(x)
            # Send the result to the server
            print("Sending intermediate result to the server")
            server_data = self.client.send_data({"byte_data": serialized_data, "stage": "forward"})["byte_data"]
            x = pickle.loads(server_data)

            if type(server_data) is str:
                break

            layer_index += 1
        return x

    def backward(self, loss: torch.Tensor):
        """Compute backward result of all model layers."""

        print("Start backward propagation")
        # iterate all layers in reverse order
        layer_index = len(self.layers) - 1

        # last forward result
        last_forward_result = self.forward_results.pop()
        last_forward_result.grad = torch.autograd.grad(outputs=loss, inputs=last_forward_result)[
            0]  # first output is the result
        self.optimizers[layer_index].step()
        self.optimizers[layer_index].zero_grad()

        # preparing grad for cloud layers
        forward_result = self.forward_results.pop()
        forward_result.grad = torch.autograd.grad(outputs=last_forward_result,
                                                  inputs=forward_result,
                                                  grad_outputs=torch.ones_like(last_forward_result),
                                                  allow_unused=True)[0]

        # Send the result back to the server
        serialized_data = pickle.dumps(forward_result.grad)
        print("Sending first grads result to the server")
        serialized_data = self.client.send_data({"byte_data": serialized_data, "stage": "backward"})["byte_data"]
        layer_index -= 1

        while layer_index >= 0:  # not the first layer, don't need to calculate it for the first layer
            # Next layer
            last_forward_result = self.forward_results.pop()
            last_forward_result.grad = pickle.loads(serialized_data)

            forward_result = self.forward_results.pop()

            self.optimizers[layer_index].step()
            self.optimizers[layer_index].zero_grad()

            if layer_index != 0:  # not the first layer, don't need to calculate it for the first layer
                forward_result.grad = torch.autograd.grad(outputs=last_forward_result,
                                                          inputs=forward_result,
                                                          grad_outputs=torch.ones_like(last_forward_result),
                                                          allow_unused=True)[0]
                # Send the result back to the client
                serialized_data = pickle.dumps(forward_result.grad)
                print("Sending intermediate grads result to the server")
                serialized_data = self.client.send_data({"byte_data": serialized_data})["byte_data"]

            layer_index -= 1

        self.forward_results = []

    def model_train(self, dataloader: DataLoader, epochs: int, device=None):
        """
        Train the network on the training set.
        """
        self.to(self.device)
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

        for epoch in range(base_epoch, base_epoch + epochs):
            loss = None
            for i, (inputs, labels) in enumerate(dataloader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # while True:
                #     try:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                self.backward(loss)
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                print("________________________________________________")
                # break
                # except NoneException as e:
                #     print(e)
                #     print("Passive repeat")
                # except Exception as e:
                #     print(e)
                #     self.socket.send_data(b"")
                #     print("Active repeat")

            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            self.save_local(epoch, loss, optimizer.state_dict())

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO: Implement this method
        pass

    def process(self, query):
        result = None
        while True:
            # pickle the tensor data
            serialized_data = pickle.dumps(query)
            # Send the result to the server
            print("Sending query to the server")
            server_data = self.client.send_data({"byte_data": serialized_data, "stage": "forward"})["byte_data"]
            # print(repr(serialized_data))
            x = pickle.loads(server_data)

            if x is None:
                break
            elif type(x) is torch.Tensor:
                x = x.to(self.device)
                x = self.forward(x)
            # else: (type(x) is str)

            result = x
        return result
