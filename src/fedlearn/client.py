from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl

from src.data import AbstractData, CifarData
from src.model import AbstractModel, DemoModel
from src.comn import AbstractClient

class Client(fl.client.NumPyClient, AbstractClient):
    def __init__(self, data: AbstractData, model: AbstractModel):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch_num = 1

        self.model = model
        self.trainloader, self.testloader = data.trainloader, data.testloader
        self.num_examples = data.get_number_examples()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config={}):
        self.set_parameters(parameters)
        self.model.model_train(self.trainloader, self.epoch_num, self.device)
        return self.get_parameters(config), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.model_test(self.testloader, self.device)
        print(f"loss: {loss}, accuracy{accuracy}") # TODO refactor to log
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def run(self):
        fl.client.start_numpy_client(
            server_address="localhost:8080", client=self)

if __name__ == '__main__':
    CLIENT_DIR = "../../tmp/client/c02"

    # Init data and model.
    data = CifarData(data_dir=CLIENT_DIR)
    model = DemoModel(None, model_dir=CLIENT_DIR)

    client = Client(data, model)
    client.run()



