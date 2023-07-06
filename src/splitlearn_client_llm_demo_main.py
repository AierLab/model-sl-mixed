from collections import OrderedDict

import torch.nn
from transformers import AutoTokenizer, AutoModel

from data import CifarData
from model import ChatModel
from splitlearn import SplitClient
import torch.nn as nn

from model.model_split_client import SplitClientModel

if __name__ == '__main__':
    # run in separate terminal
    # CLIENT_DIR = "../tmp/client/c01"
    CLIENT_DIR = "../tmp/client/c02"
    # CLIENT_DIR = "../tmp/client/c01"

    client = SplitClient('http://localhost:8888', 'secret_api_key')

    chat_model = ChatModel(CLIENT_DIR, client=client)

    print("Welcome to the ChatGLM-6B model. Type your message.")

    while True:
        query = input("\nUser: ").strip()
        if query == "stop":
            chat_model.stop()
            break
        elif query == "clear":
            chat_model.clear()
            print("Chat history cleared.")
        else:
            response = chat_model.forward(query)
            print(response)
