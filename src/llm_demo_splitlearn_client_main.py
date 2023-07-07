from data import CifarData
from splitlearn import SplitClient
import torch.nn as nn

from model.model_split_client import SplitClientModel

layer_num = 28

if __name__ == '__main__':
    # run in separate terminal
    # CLIENT_DIR = "../tmp/client/c01"
    CLIENT_DIR = "../tmp/client/c02"
    # CLIENT_DIR = "../tmp/client/c01"

    model_layers = nn.ModuleList([nn.Linear(in_features=4096, out_features=4096, bias=True) for layer_id in range(layer_num) if layer_id % 7 == 0])

    # Init data, socket and model.
    data = CifarData(data_dir=CLIENT_DIR)
    client = SplitClient('http://localhost:10086', 'secret_api_key')

    model = SplitClientModel(model_layers, client, CLIENT_DIR)
    model.model_train(data.trainloader, epochs=1)
