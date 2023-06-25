import os

from .socket_abstract import AbstractSocket

import socket


class ClientSocket(AbstractSocket):
    def __init__(self, host: str, port: int):
        super().__init__()

        try:
            self.socket.connect((host, port))
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return
