import socket


class ClientSocket:
    def __init__(self, host: str, port: int):
        # Create a TCP/IP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.client_socket.connect((host, port))
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return

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
                print(repr(chunk))
                data.append(chunk)
            data = b"".join(data)
            # print(repr(data))
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")

    # def send_file(self, file_path):
    #     if not os.path.exists(file_path):
    #         print(f"File {file_path} does not exist")
    #         return
    #     try:
    #         # Open the file in binary mode
    #         with open(file_path, 'rb') as f:
    #             # Loop over the file
    #             while True:
    #                 data = f.read(1024)  # read file by chunks of size 1024 bytes
    #                 print(f"Sending {data}")
    #                 if not data:
    #                     print("EOF reached")
    #                     self.client_socket.sendall(b"EOF")
    #                     break  # end of file reached
    #                 self.client_socket.sendall(data)
    #     except Exception as e:
    #         print(f"Error sending file: {e}")
    #
    # def receive_file(self, file_path):
    #     try:
    #         # Create (or overwrite) a file and write the incoming data into it
    #         with open(file_path, 'wb') as f:
    #             while True:
    #                 data = self.client_socket.recv(1024)  # receive 1024 bytes
    #                 if not data:
    #                     break  # no more data
    #                 f.write(data)
    #     except Exception as e:
    #         print(f"Error receiving file: {e}")

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
