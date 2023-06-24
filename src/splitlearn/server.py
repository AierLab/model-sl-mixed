from comn import AbstractServer
from model import AbstractModel
import socket
import os
import torch

class SplitServer(AbstractServer):
    def __init__(self, model: AbstractModel, epoch_num: int, ckpt_path: str, host: str, port: int):
        self.model = model
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.host = host
        self.port = port
        self.epoch_num = epoch_num
        self.model.to(self.device)

        # Create a TCP/IP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.host, self.port)
        
        print(f"Starting server on {self.host}:{self.port}")
        try:
            self.server_socket.bind(server_address)
        except Exception as e:
            print(f"Could not start server: {e}")
            return

    def send_file(sock, filename):
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            return

        # Open the file in binary mode
        with open(filename, 'rb') as f:
            # Loop over the file
            for data in f:
                # Send the data over the socket
                sock.sendall(data)

    def get_socket(self):
        return self.server_socket
    
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))
    
    def run(self):
        # Listen for incoming connections
        self.server_socket.listen(1)
        while True:
            # Wait for a connection
            print("Waiting for a connection...")
            connection, client_address = self.server_socket.accept()
            try:
                print(f"Connection from {client_address}")

                # Receive the data in small chunks and retransmit it
                
                while True:
                    data = connection.recv(1024)
                    print(f"Received {data}")
                    if data:
                        print("Sending data back to the client")
                        connection.sendall(data)
                    else:
                        print("No more data from", client_address)
                        break
            finally:
                # Clean up the connection
                connection.close()
                print("Connection closed")

    

