
import socket
import os

from .socket_abstract import AbstractSocket


class ServerSocket(AbstractSocket):
    def __init__(self, host: str, port: int):
        super().__init__()

        server_address = (host, port)
        print(f"Starting server on {host}:{port}")
        try:
            self.socket.bind(server_address)
            # Listen for incoming connections
            self.socket.listen(1)
            print('waiting for a connection')
            self.client_socket, self.client_address = self.socket.accept()
            print('client connected:', self.client_address)
        except Exception as e:
            print(f"Could not start server: {e}")
            return
