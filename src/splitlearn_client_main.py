from data import CifarData
from comn import ClientSocket
import torch.nn as nn

from model.model_split_client import SplitClientModel
from splitlearn import SplitClient

if __name__ == '__main__':
    # run in separate terminal
    # CLIENT_DIR = "../tmp/client/c01"
    CLIENT_DIR = "../tmp/client/c02"
    # CLIENT_DIR = "../tmp/client/c01"

    m1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

    m3 = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(8 * 8 * 64, 64),
        nn.Linear(64, 32),
        nn.Linear(32, 10))

    model_layers = nn.Sequential(m1, m3)

    # Init data, socket and model.
    data = CifarData(data_dir=CLIENT_DIR)
    socket = ClientSocket(host="localhost", port=10086)
    model = SplitClientModel(model_layers, socket, CLIENT_DIR)

    client = SplitClient(data, model)
    client.run()
