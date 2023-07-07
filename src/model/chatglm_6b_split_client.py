import pickle
from time import sleep

import numpy as np
import torch.nn
from torch import nn
from transformers import AutoTokenizer, AutoModel

from model import AbstractModel
from .layer_split_server import SplitServerLayer
from .layer_split_client import SplitClientLayer

layer_number = 28


class SplitClientChatModel(AbstractModel):
    def __init__(self, model_dir, client):
        super().__init__(model_dir)
        self.model_dir = model_dir
        self.client = client

        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm_6b_demo", trust_remote_code=True)
        self.model: torch.nn.Module = AutoModel.from_pretrained("THUDM/chatglm_6b_demo", trust_remote_code=True)

        # Replace transformer layers with new layer.
        for i in range(layer_number - 1, 0, -1):
            self.model.transformer.layers[i] = SplitClientLayer(client, model_dir)

        self.model = self.model.half().cuda()

        print(self.model)
        self.model_eval = self.model.eval()

        self.history = []
        self.count = 0

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.model(**x)
        return x

    def _build_prompt(self) -> str:
        # prompt = "Welcome to the ChatGLM-6B model. Type your message."
        prompt = ""
        _, response = self.history[-1]
        prompt += response
        return prompt

    def process(self, query) -> str:
        if self.count == 1000:  # TODO hard coded
            self.count = 0
            self.history = []
        for response, self.history in self.model_eval.stream_chat(self.tokenizer, query, history=self.history):
            self.count += 1
            # if count % 8 == 0:
            #     yield self.build_prompt()
        return self._build_prompt()

    def clear(self) -> None:
        self.history = []

    def stop(self) -> None:
        self.model = None

    def run(self):
        """
        Train the network on the training set.
        """

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])

        print("Welcome to the ChatGLM-6B model. Type your message.")
        while True:
            query = input("\nUser: ").strip()
            if query == "stop":
                self.stop()
                print("Process exit..")
                break
            elif query == "clear":
                self.clear()
                print("Chat history cleared.")
            else:
                response = self.process(query)
                print(response)


def main():
    chat_model = SplitClientChatModel()
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
            response = chat_model.process(query)
            print(response)

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm_6b_demo", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm_6b_demo", trust_remote_code=True).half().cuda()
    model = model.half().cuda()
    model.eval()

    # Inference
    input = tokenizer(
        "Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or "
        "SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.",
        return_tensors="pt")
    input = input.to('cuda')
    outputs = model.generate(**input, max_length=512)
    print(tokenizer.decode(outputs[0].tolist()))

    # Training
    inputs = tokenizer(
        ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
        return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')
    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits


if __name__ == "__main__":
    main()
