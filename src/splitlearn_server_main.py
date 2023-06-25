import torch.nn as nn

from data import CifarData
from comn import ServerSocket
from model.model_split_server import SplitServerModel
from splitlearn import SplitClient, SplitServer

if __name__ == "__main__":
    m2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    model_layers = nn.Sequential(m2)

    SERVER_DIR = "../tmp/server"

    # Init data and model.
    socket = ServerSocket(host="localhost", port=10086)
    model = SplitServerModel(model_layers, socket, SERVER_DIR)

    server = SplitServer(model)
    server.run()
