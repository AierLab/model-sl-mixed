import socket
import os


class ServerSocket:
    def __init__(self, host: str, port: int):
        # Create a TCP/IP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_address = (host, port)
        print(f"Starting server on {host}:{port}")
        try:
            self.server_socket.bind(server_address)
            # Listen for incoming connections
            self.server_socket.listen(1)
        except Exception as e:
            print(f"Could not start server: {e}")

    def send_data(self, data: bytes):
        """Sends raw data through the socket."""
        
        print('waiting for a connection')
        client_socket, client_address = self.server_socket.accept()
        print('client connected:', client_address)
        try:
            client_socket.sendall(data + b"EOF")
        except Exception as e:
            print(f"Error sending data: {e}")
        finally:
            print(f"client deleted")
            client_socket.close()

    def receive_data(self) -> bytes:
        """Receives raw data from the socket."""
        print('recv: waiting for a connection')
        client_socket, client_address = self.server_socket.accept()
        print('client connected:', client_address)

        data = []
        try:
            while True:
                chunk = client_socket.recv(4096)
                if b"EOF" in chunk:
                    data.append(chunk[:-3])
                    print("EOF received")
                    break  # no more data
                data.append(chunk)
            data = b"".join(data)
            print(repr(data))
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")
        finally:
            print(f"client deleted")
            client_socket.close()

    def send_file(self, file_path):
        print('waiting for a connection')
        client_socket, client_address = self.server_socket.accept()
        print('client connected:', client_address)
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
                    client_socket.sendall(data)
        except Exception as e:
            print(f"Error sending file: {e}")
        finally:
            client_socket.close()

    def receive_file(self, file_path):
        print('waiting for a connection')
        client_socket, client_address = self.server_socket.accept()
        print('client connected:', client_address)
        try:
            # Create (or overwrite) a file and write the incoming data into it
            with open(file_path, 'wb') as f:
                while True:
                    data = client_socket.recv(1024)  # receive 1024 bytes
                    if not data:
                        break  # no more data
                    f.write(data)
        except Exception as e:
            print(f"Error receiving file: {e}")
        finally:
            client_socket.close()
