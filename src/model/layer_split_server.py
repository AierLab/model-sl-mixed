from time import sleep
import torch
from model import AbstractModel
from queue import Queue


class SplitServerLayer(AbstractModel):
    def __init__(self, model_dir: str, in_queue: Queue, out_queue: Queue, first_layer=False, last_layer=False):
        super().__init__(model_dir)
        # get all model layers
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.first_layer = first_layer
        self.last_layer = last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.first_layer:
            while not self.out_queue.empty():
                sleep(0.1)
            self.out_queue.put(x)
        while self.in_queue.empty():
            sleep(0.1)
        x = self.in_queue.get()
        return x

    def backward(self, grad: torch.Tensor):
        if not self.last_layer:
            while not self.out_queue.empty():
                sleep(0.1)
            self.out_queue.put(grad)
        while self.in_queue.empty():
            sleep(0.1)
        grad = self.in_queue.get()
        return grad
