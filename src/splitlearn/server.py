"""
This script is used to define a Server class for a Flask server.
"""
import base64

from flask import Flask, request, jsonify, abort
from queue import Queue
from threading import Thread
from typing import Callable, Any, Optional
import time


class SplitServer:
    """
    This class defines a Server that can receive, process, and send data.
    """

    def __init__(self, process_func: Callable) -> None:
        """
        Initializer for the Server class.
        """
        self.app = Flask(__name__)
        self.app.add_url_rule('/intermediate', 'intermediate', self.receive_data, methods=['POST'])
        self.api_key = "secret_api_key"  # In reality, this should be securely stored and not hard-coded
        self.data_queue = Queue()
        self.processed_data = None
        self.status = "waiting"
        self.process_func = process_func

    def check_auth(self) -> bool:
        """
        Checks if the API key is present and correct.
        """
        return request.headers.get('X-API-Key') == self.api_key

    def receive_data(self) -> Any:
        """
        Receives data and pushes it into the data_queue.
        """
        if not self.check_auth():
            abort(401)  # Unauthorized
        data = request.json
        for key in data:
            if "byte" in key:
                data[key] = base64.b64decode(data[key].encode('utf-8'))
        self.data_queue.put(data)
        self.status = "processing"
        print(f"Received data.")
        data = self.process_func(data)
        for key in data:
            if "byte" in key:
                data[key] = base64.b64encode(data[key]).decode('utf-8')
        return jsonify(data)

    def run(self, host: str, port: int) -> None:
        """
        Runs the Flask server.
        """
        self.app.run(host=host, port=port)


if __name__ == '__main__':
    def process_func(data: dict) -> dict:
        """
        Custom processing function.
        """
        # time.sleep(0.5)  # Simulate processing time
        return data

    server = SplitServer(process_func)
    server.run("localhost", 10086)
