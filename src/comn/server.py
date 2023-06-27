import socket


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
            print('recv: waiting for a connection')
            self.client_socket, client_address = self.server_socket.accept()
            print('client connected:', client_address)
        except Exception as e:
            print(f"Could not start server: {e}")

    def send_data(self, data: bytes):
        """Sends raw data through the socket."""
        try:
            self.client_socket.sendall(data + b"EOF")
        except Exception as e:
            print(f"Error sending data: {e}")

    def receive_data(self) -> bytes:
        """Receives raw data from the socket."""
        data = []
        try:
            while True:
                chunk = self.client_socket.recv(4096)
                if b"EOF" in chunk:
                    data.append(chunk[:-3])
                    break  # no more data
                # print(repr(chunk))
                data.append(chunk)
            data = b"".join(data)
            # print(repr(data))
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
