from comn import AbstractClient
from model import AbstractModel
from data import AbstractData
from model.model_split_client import SplitClientModel
import torch
import socket


class SplitClient(AbstractClient):
    def __init__(self, data: AbstractData, model: SplitClientModel, host: str, port: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch_num = 1
        self.host = host
        self.port = port

        self.model = model
        self.trainloader, self.testloader = data.trainloader, data.testloader
        self.num_examples = data.get_number_examples()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def set_parameters(self, ckpt_path):
    #     self.model.load_state_dict(torch.load(ckpt_path))

    def fit(self, config):
        self.model.model_train(self.trainloader, self.epoch_num, self.device)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, config):
        loss, accuracy = self.model.model_test(self.testloader, self.device)
        print(f"loss: {loss}, accuracy{accuracy}")  # TODO refactor to log
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
    def get_socket(self):
        return self.client_socket
    
    def run(self):
        # Create a TCP/IP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
        try:
            self.client_socket.connect((self.host, self.port))
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return
        
        self.model.set_socket(self.client_socket)
    
        

