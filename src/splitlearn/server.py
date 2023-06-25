from comn import AbstractServer
from model import AbstractModel
from model.model_split_server import SplitServerModel

import socket
import os
import torch


class SplitServer(AbstractServer):
    def __init__(self, model: SplitServerModel, epoch_num: int, ckpt_path: str, host: str, port: int):
        self.model = model
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.host = host
        self.port = port
        self.epoch_num = epoch_num
        self.model.to(self.device)

    # def send_file(sock, filename):
    #     if not os.path.exists(filename):
    #         print(f"File {filename} does not exist")
    #         return

    #     # Open the file in binary mode
    #     with open(filename, 'rb') as f:
    #         # Loop over the file
    #         for data in f:
    #             # Send the data over the socket
    #             sock.sendall(data)

    def get_socket(self):
        return self.server_socket

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def set_parameters(self):
    #     self.model.load_state_dict(torch.load(self.ckpt_path))

    def fit(self):
        self.model.model_train(self.epoch_num, self.device)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def run(self):
        # Create a TCP/IP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.host, self.port)

        print(f"Starting server on {self.host}:{self.port}")
        try:
            self.server_socket.bind(server_address)
        except Exception as e:
            print(f"Could not start server: {e}")
            return

        self.model.set_socket(self.server_socket)
