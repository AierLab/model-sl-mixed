import pickle
from time import sleep
from typing import Collection, Tuple

import torch
from model import AbstractModel
from queue import Queue


class SplitServerLayer(AbstractModel):
    def __init__(self, model_dir: str, in_queue: Queue, out_queue: Queue, device=None, skip=False, first_layer=False, last_layer=False):
        super().__init__(model_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # get all model layers
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.skip = skip
        self.first_layer = first_layer
        self.last_layer = last_layer

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        if self.skip:
            return x, kwargs

        if not self.first_layer:
            while not self.out_queue.empty():
                sleep(0.1)
            kwargs.update({"data_down": x})
            serialized_data = pickle.dumps(kwargs)
            data = {"byte_data": serialized_data, "stage": "forward"}
            # Send the result to the server
            self.out_queue.put(data)

        while self.in_queue.empty():
            sleep(0.1)
        data = self.in_queue.get()
        # print(repr(serialized_data))
        kwargs = pickle.loads(data["byte_data"])
        x = kwargs.pop("data_up")
        x = x.to(self.device)

        return x, kwargs

    def backward(self, grad: torch.Tensor):
        if not self.last_layer:
            while not self.out_queue.empty():
                sleep(0.1)
            # pickle the tensor data
            serialized_data = pickle.dumps(grad)
            # Send the result to the server
            data = {"byte_data": serialized_data, "stage": "forward"}
            self.out_queue.put(data)

        while self.in_queue.empty():
            sleep(0.1)
        data = self.in_queue.get()
        # print(repr(serialized_data))
        grad = pickle.loads(data["byte_data"])
        x = grad.to(self.device)

        return x
