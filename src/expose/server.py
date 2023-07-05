from flask import Flask, request, jsonify, abort
from typing import Callable, Any


class Server:
    """
    A Server class to handle text processing and feedback reception via a simple Flask-based API.
    """

    def __init__(self, function: Callable[[str], str]):
        """
        Initialize the Server.

        :param function: The function to be used for processing text.
        """
        self.app = Flask(__name__)
        self.app.route("/api/text", methods=['POST'])(self.post_text)
        self.app.route("/api/feedback", methods=['POST'])(self.post_feedback)
        self.api_key = "secret_api_key"  # In reality, this should be securely stored
        self.function = function

    def check_auth(self) -> bool:
        """
        Check the request for correct authorization.

        :return: Whether the request is authorized.
        """
        return request.headers.get('X-API-Key') == self.api_key

    def post_text(self) -> Any:
        """
        Handle a POST request to the /api/text endpoint.

        :return: A JSON response containing the result of processing the request's text.
        """
        if not self.check_auth():
            abort(401)  # Unauthorized
        data = request.get_json()
        text = data.get('text')
        user_feedback = data.get('user_feedback')
        resource_requirements = data.get('resource_requirements')

        if text:
            result = self.function(text)  # Process the text
            return jsonify({"result": result}), 200
        else:
            return jsonify({"error": "Invalid request parameter"}), 400

    def post_feedback(self) -> Any:
        """
        Handle a POST request to the /api/feedback endpoint.

        :return: A JSON response confirming receipt of feedback.
        """
        if not self.check_auth():
            abort(401)  # Unauthorized
        data = request.get_json()
        feedback = data.get('feedback')

        if feedback:
            return jsonify({"message": "Feedback received"}), 200
        else:
            return jsonify({"error": "Invalid request parameter"}), 400

    def run(self, host: str, port: int) -> None:
        """
        Runs the Flask server.
        """
        self.app.run(host=host, port=port)


if __name__ == '__main__':
    def process_func(data: str) -> str:
        """
        Example text processing function.

        :param data: The text to be processed.
        :return: The processed text.
        """
        return data  # In reality, this would be replaced by actual processing logic


    server = Server(process_func)
    server.run("localhost", 10086)
