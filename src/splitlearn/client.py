"""
This script is used to define a Client class that communicates with a server.
"""
import base64

import requests
import time
from typing import Any, Dict


class SplitClient:
    """
    This class defines a Client that can send data to a server, wait for processing, and receive processed data.
    """

    def __init__(self, url: str, api_key: str) -> None:
        """
        Initializer for the Client class.

        Args:
            url: The URL of the server.
            api_key: The API key for authentication.
        """
        self.url = url
        self.api_key = api_key
        self.headers = {'X-API-Key': self.api_key}

    def send_data(self, data: Dict[str, Any]) -> None:
        """
        Sends data to the server.

        Args:
            data: The data to send.
        """
        for key in data:
            if "byte" in key:
                data[key] = base64.b64encode(data[key]).decode('utf-8')
        response = requests.post(self.url + "/intermediate", json=data, headers=self.headers)
        data = response.json()
        for key in data:
            if "byte" in key:
                data[key] = base64.b64decode(data[key].encode('utf-8'))
        print(f"Received processed data.")
        return data

if __name__ == '__main__':
    client = SplitClient('http://localhost:10086', 'secret_api_key')
    client.send_data({"byte_data": b"value"})
