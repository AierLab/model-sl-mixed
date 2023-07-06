from queue import Queue
from threading import Thread

import torch.nn as nn

from splitlearn import SplitServer
from model.model_split_server import SplitServerModel

if __name__ == "__main__":
    m2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    model_layers = nn.ModuleList([m2])

    SERVER_DIR = "../tmp/server"

    in_queue = Queue()
    out_queue = Queue()

    # Init data and model.
    model = SplitServerModel(model_layers, SERVER_DIR, in_queue, out_queue)
    server = SplitServer(in_queue, out_queue)

    t1 = Thread(target=model.run)
    t2 = Thread(target=server.run, args=("localhost", 8888))

    t1.start()
    t2.start()
