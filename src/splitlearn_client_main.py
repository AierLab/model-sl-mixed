from torch import nn

from data import CifarData
from model import SplitClientModel
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

    m2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    m3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    model_layers = nn.Sequential(m1, m2, m3)

    # Init data and model.
    data = CifarData(data_dir=CLIENT_DIR)
    model = SplitClientModel(CLIENT_DIR, model_layers)

    client = SplitClient(data, model, host="localhost", port=10086)
    client.run()
    client.fit(None)
