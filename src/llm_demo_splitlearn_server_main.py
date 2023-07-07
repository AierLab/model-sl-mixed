from collections import OrderedDict
from queue import Queue
from threading import Thread

import torch.nn
from transformers import AutoTokenizer, AutoModel
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from model.chatglm_6b_split_server import ChatGLMForConditionalGeneration, ChatGLMTokenizer, ChatGLMConfig
from splitlearn import SplitClient, SplitServer

tokenizer_config_path = os.path.join('model', 'chatglm_6b_demo', 'tokenizer_config.json')
model_config_path = os.path.join('model', 'chatglm_6b_demo', 'config.json')
token_text_path = os.path.join("..", "tmp", "client", "ice_text.model")
model_state_dict_file_num = 8

class ChatModel:
    def __init__(self, model_dir: str = None, in_queue: Queue = None, out_queue: Queue = None): # FIXME FOR TEST USE ONLY, can not be none
        if model_dir: # FIXME FOR TEST USE ONLY, never load from server path
            model_dir = os.path.join("..", "tmp", "server")

        # Open and load the JSON file into a Python dict
        with open(tokenizer_config_path) as config_file:
            config_dict = json.load(config_file)

        self.tokenizer = ChatGLMTokenizer(token_text_path, **config_dict)
        # self.model = AutoModel.from_pretrained("THUDM/chatglm_6b_demo", trust_remote_code=True).half().cuda()

        # Open and load the JSON file into a Python dict
        with open(model_config_path) as config_file:
            config_dict = json.load(config_file)

        configuration = ChatGLMConfig(**config_dict)
        model = ChatGLMForConditionalGeneration(configuration, model_dir, in_queue, out_queue)

        # Empty dict to accumulate the state dicts from all files
        state_dict_all = {}

        # Loop over files
        for i in tqdm(range(1, model_state_dict_file_num + 1)):
            filename = f"pytorch_model-{str(i).zfill(5)}-of-{str(model_state_dict_file_num).zfill(5)}.bin"

            filepath = os.path.join(model_dir, filename)  # replace with the directory of the files
            state_dict = torch.load(filepath)
            state_dict_all.update(state_dict)

        # Load the combined state_dict into the model
        model.load_state_dict(state_dict_all, strict=False)

        self.model = model.half().cuda()

        # print(self.model)
        self.model = self.model.eval()
        self.history = []
        self.count = 0

    def build_prompt(self) -> str:
        # prompt = "Welcome to the ChatGLM-6B model. Type your message."
        prompt = ""
        _, response = self.history[-1]
        prompt += response
        return prompt

    def process(self, query: str) -> str:
        if self.count == 1000:  # TODO hard coded
            self.count = 0
            self.history = []
        for response, self.history in self.model.stream_chat(self.tokenizer, query, history=self.history):
            self.count += 1
            # if count % 8 == 0:
            #     yield self.build_prompt()
        return self.build_prompt()

    def clear(self) -> None:
        self.history = []

    def stop(self) -> None:
        self.model = None


def main():
    SERVER_DIR = "../tmp/server"

    in_queue = Queue()
    out_queue = Queue()

    # Init data and model.
    model = ChatModel(SERVER_DIR, in_queue, out_queue)
    server = SplitServer(in_queue, out_queue)

    def run():
        while True:
            print("Welcome to the ChatGLM-6B model. Type your message.")
            response = model.process("")
            print(response)

    t1 = Thread(target=run)
    t2 = Thread(target=server.run, args=("localhost", 8888))

    t1.start()
    t2.start()

if __name__ == "__main__":
    main()
