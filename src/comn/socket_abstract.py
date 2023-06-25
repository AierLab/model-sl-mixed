import os
from abc import ABC, abstractmethod
import socket


class AbstractSocket(ABC):
    def __init__(self):
        # Create a TCP/IP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send_data(self, data: bytes):
        """Sends raw data through the socket."""
        try:
            self.socket.sendall(data)
        except Exception as e:
            print(f"Error sending data: {e}")

    def receive_data(self) -> bytes:
        """Receives raw data from the socket."""
        data = None
        try:
            while True:
                chunk = self.socket.recv(1024)
                if not chunk and data:
                    break  # no more data
                data += chunk
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")

    def send_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            return
        try:
            # Open the file in binary mode
            with open(file_path, 'rb') as f:
                # Loop over the file
                while True:
                    data = f.read(1024)  # read file by chunks of size 1024 bytes
                    if not data:
                        break  # end of file reached
                    self.socket.sendall(data)
        except Exception as e:
            print(f"Error sending file: {e}")

    def receive_file(self, file_path):
        try:
            # Create (or overwrite) a file and write the incoming data into it
            with open(file_path, 'wb') as f:
                while True:
                    data = self.socket.recv(1024)  # receive 1024 bytes
                    if not data:
                        break  # no more data
                    f.write(data)
        except Exception as e:
            print(f"Error receiving file: {e}")

    def close_connection(self):
        if self.socket:
            self.socket.close()
