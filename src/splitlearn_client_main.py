from data import CifarData
from splitlearn import SplitClient
import torch.nn as nn

from model.model_split_client import SplitClientModel

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
        nn.Linear(32, 10)
    )

    model_layers = nn.ModuleList([m1, m3])

    # Init data, socket and model.
    data = CifarData(data_dir=CLIENT_DIR)
    client = SplitClient('http://localhost:10086', 'secret_api_key')

    model = SplitClientModel(model_layers, client, CLIENT_DIR)
    model.model_train(data.trainloader, epochs=1)
