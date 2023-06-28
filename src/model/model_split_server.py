import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from comn import ServerSocket
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
    def __init__(self, model_layers, socket: ServerSocket, model_dir: str):
        super().__init__(model_dir)
        # get all model layers
        self.layers = nn.ModuleList(list(model_layers.children()))
        self.socket = socket
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.forward_results: List[torch.Tensor] = []
        self.optimizers = [Adam(layer.parameters(), lr=0.001, ) for layer in self.layers]

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate all layers
        layer_index = 0
        self.forward_results = []
        while layer_index < len(self.layers):
            # receive the result from the server
            print("Waiting forward intermediate result from the client")
            server_data = self.socket.receive_data()  # Server receive data first
            x = pickle.loads(server_data)
            #print("data recvd : " + repr(x))
            x = x.to(self.device)

            # # Save the input tensor to a local file # FIXME never used, may need to be removed
            # torch.save(x, f'../tmp/client/{type(self).__name__}/layer_{layer_index}_input.pt')

            # Compute the forward result of the tensor data
            self.forward_results.append(x)
            x = self.layers[layer_index](x)
            self.forward_results.append(x)

            # pickle the tensor data
            serialized_data = pickle.dumps(x)
            # # Save the tensor data to a local file # FIXME never used, may need to be removed
            # torch.save(serialized_data, f'../tmp/client/{type(self).__name__}/layer_{layer_index}_output.pt')

            # Send the result to the server
            print("Sending forward intermediate result to the client")
            self.socket.send_data(serialized_data)
            layer_index += 1
        return x

    def backward(self):
        """Compute backward result of all model layers."""
        # iterate all layers in reverse order
        layer_index = len(self.layers) - 1

        while layer_index >= 0:
            last_forward_result = self.forward_results.pop()

            print("Waiting intermediate grads from the client")
            serialized_data = self.socket.receive_data()
            last_forward_result.grad = pickle.loads(serialized_data)

            forward_result = self.forward_results.pop()
            forward_result.grad = torch.autograd.grad(outputs=last_forward_result,
                                                      inputs=forward_result,
                                                      grad_outputs=torch.ones_like(last_forward_result),
                                                      allow_unused=True)[0]
            # TODO do not training on server side.
            # self.optimizers[layer_index].step()
            # self.optimizers[layer_index].zero_grad()

            # Send the result back to the client
            serialized_data = pickle.dumps(forward_result.grad)
            print("Sending intermediate grads to the client")
            self.socket.send_data(serialized_data)

            layer_index -= 1

            # # Save the tensor data to a local file # FIXME never used, may need to be removed
            # torch.save(tensor_data, f'../tmp/client/layer_{layer_index}_grads_input.pt')
            # print("Sending intermediate result back to the client")
            # Compute the backward result of the tensor data

    def model_train(self, device: torch.device, dataloader: DataLoader = None, epochs: int = None):
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
            try:
                self.forward()
                self.backward()
                # print("________________________________________________")
            except NoneException as e:
                print(e)
                print("Passive repeat")
            except Exception as e:
                print(e)
                self.socket.send_data(b"")
                print("Active repeat")

    def model_test(self, dataloader: DataLoader, device: torch.device = None) -> Tuple[float, float]:
        """
        Validate the network on the entire test set.
        return
        loss, accuracy
        """
        # TODO: Implement this method
        pass
