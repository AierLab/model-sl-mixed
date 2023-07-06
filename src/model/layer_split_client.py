from splitlearn import SplitClient
import pickle
from typing import List

import torch.nn as nn
import torch
from torch.optim import Adam
from model import AbstractModel


class SplitClientLayer(AbstractModel):
    def __init__(self, client: SplitClient, model_dir: str, device=None, first_layer=False, last_layer=False):
        super().__init__(model_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.client = client
        self.forward_results: List[torch.Tensor] = []

        # TODO can have more layers
        self.linear = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.optimizer = Adam(self.linear.parameters(), lr=0.001)

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward result of all model layers."""
        # iterate all layers
        layer_index = 0
        self.forward_results = []

        # Compute the forward result of the tensor data
        self.forward_results.append(x)
        x = self.linear(x)
        self.forward_results.append(x)

        # Not communicate with server in the end of inference.
        if self.last_layer:
            return x

        # pickle the tensor data
        serialized_data = pickle.dumps(x)
        # Send the result to the server
        print("Sending intermediate result to the server")
        server_data = self.client.send_data({"byte_data": serialized_data, "stage": "forward"})["byte_data"]
        # print(repr(serialized_data))
        x = pickle.loads(server_data)
        x = x.to(self.device)
        return x

    def backward(self, grad_input: torch.Tensor = None, loss: torch.Tensor = None):
        """Compute backward result of all model layers."""

        print("Start backward propagation")

        # last forward result
        if self.last_layer:
            last_forward_result = self.forward_results.pop()
            last_forward_result.grad = torch.autograd.grad(outputs=loss, inputs=last_forward_result)[
                0]  # first output is the result
            self.linear.step()
            self.linear.zero_grad()

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
            grad = pickle.loads(serialized_data)
            grad.to(self.device)
            return grad
        else:
            last_forward_result = self.forward_results.pop()
            last_forward_result.grad = grad_input

            forward_result = self.forward_results.pop()

            self.linear.step()
            self.linear.zero_grad()

            if not self.first_layer:
                forward_result.grad = torch.autograd.grad(outputs=last_forward_result,
                                                          inputs=forward_result,
                                                          grad_outputs=torch.ones_like(last_forward_result),
                                                          allow_unused=True)[0]
                # Send the result back to the client
                serialized_data = pickle.dumps(forward_result.grad)
                print("Sending intermediate grads result to the server")
                serialized_data = self.client.send_data({"byte_data": serialized_data})["byte_data"]
                grad = pickle.loads(serialized_data)
                grad.to(self.device)
                return grad
