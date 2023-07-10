from data import CifarData
from expose import Server
from splitlearn import SplitClient
import torch.nn as nn

from model import SplitClientModel

layer_num = 28

if __name__ == '__main__':
    # run in separate terminal
    # CLIENT_DIR = "../tmp/client/c01"
    CLIENT_DIR = "../tmp/client/c02"
    # CLIENT_DIR = "../tmp/client/c01"

    # model_layers = nn.ModuleList([])
    model_layers = nn.ModuleList([nn.Identity() for layer_id in range(1)])
    # model_layers = nn.ModuleList([nn.Linear(in_features=4096, out_features=4096, bias=True) for layer_id in range(4)])

    # Init data, socket and model.
    client = SplitClient('http://localhost:8888', 'secret_api_key')
    model = SplitClientModel(model_layers, client, CLIENT_DIR).half()

    print("Welcome to the ChatGLM-6B model. Type your message.")
    print("Welcome to the ChatGLM-6B model. Type your message.")
    server = Server(model.process)
    server.run("localhost", 10086)
