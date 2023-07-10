from flask import Flask, request, jsonify, abort
from flask_cors import CORS
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
        CORS(self.app)

        self.app.route("/api/text", methods=['POST'])(self.post_text)
        self.app.route("/api/feedback", methods=['POST'])(self.post_feedback)
        self.app.route("/api/history", methods=['DELETE'])(self.delete_history)
        self.app.route("/api/history", methods=['POST'])(self.post_history)
        self.app.route("/api/chat", methods=['POST'])(self.post_chat)
        self.app.route("/api/role", methods=['POST'])(self.post_role)
        self.app.route("/api/chat/continue", methods=['POST'])(self.post_continue_chat)

        self.function = function

        # Initialize the histories and role as dictionaries
        self.histories = {}
        self.roles = {}

    def check_auth(self, role: str) -> bool:
        """
        Check the request for correct authorization.

        :return: Whether the request is authorized.
        """
        return role in self.roles

    def post_text(self) -> Any:
        """
        Handle a POST request to the /api/text endpoint.

        :return: A JSON response containing the result of processing the request's text.
        """
        data = request.get_json()
        role = data.get('role')
        if not self.check_auth(role):
            abort(401)  # Unauthorized
        text = data.get('text')

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
        data = request.get_json()
        role = data.get('role')
        if not self.check_auth(role):
            abort(401)  # Unauthorized
        feedback = data.get('feedback')

        if feedback:
            return jsonify({"message": "Feedback received"}), 200
        else:
            return jsonify({"error": "Invalid request parameter"}), 400

    def delete_history(self) -> Any:
        """
        Handle a DELETE request to the /api/history endpoint.
        Clear the chat history.
        """
        data = request.get_json()
        role = data.get('role')
        if not self.check_auth(role):
            abort(401)  # Unauthorized
        self.histories[role] = []
        return jsonify({"message": "History cleared"}), 200

    def post_history(self) -> Any:
        """
        Handle a POST request to the /api/history endpoint.
        Returns the chat history.
        """
        data = request.get_json()
        role = data.get('role')
        if not self.check_auth(role):
            abort(401)  # Unauthorized
        return jsonify({"history": self.histories[role]}), 200

    def post_chat(self) -> Any:
        """
        Handle a POST request to the /api/chat endpoint.
        Creates a new chat.
        """
        data = request.get_json()
        role = data.get('role')
        if role:
            self.roles[role] = True  # Add the role to the list of valid roles
            self.histories[role] = []  # Initialize the history for this role
            return jsonify({"message": f"New chat created with role {role}"}), 200
        else:
            return jsonify({"error": "Invalid request parameter"}), 400

    def post_role(self) -> Any:
        """
        Handle a POST request to the /api/role endpoint.
        Changes the AI role.
        """
        data = request.get_json()
        role = data.get('role')
        if role:
            self.roles[role] = True
            return jsonify({"message": f"Role changed to {role}"}), 200
        else:
            return jsonify({"error": "Invalid request parameter"}), 400

    def post_continue_chat(self) -> Any:
        """
        Handle a POST request to the /api/chat/continue endpoint.
        Continues a chat based on the history.
        """
        data = request.get_json()
        role = data.get('role')
        if not self.check_auth(role):
            abort(401)  # Unauthorized
        text = data.get('text')
        if text:
            # Process the text
            result = self.function(text)
            # Update the history
            self.histories[role].append({'sender': 'user', 'message': text})
            self.histories[role].append({'sender': 'AI', 'message': result})
            return jsonify({"response": result}), 200
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
