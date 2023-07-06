from time import sleep

import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel

from model import AbstractModel
from .layer_split_server import SplitServerLayer
from .layer_split_client import SplitClientLayer

class ChatModel(AbstractModel):
    def __init__(self, model_dir, in_queue=None, out_queue=None, client=None):
        super().__init__(model_dir)
        layer_number = 28

        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

        self.client = client

        if client is None:
            self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
            self.model.transformer.word_embeddings = nn.Identity()
            self.model.transformer.layers[0] = nn.Identity()
            for i in range(layer_number - 1, 0, -1):
                self.model.transformer.layers[i].add_module(f"p{i}_layer",
                                                            SplitServerLayer(model_dir, in_queue, out_queue))
        else:
            self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
            for i in range(layer_number - 1, 0, -1):
                self.model.transformer.layers[i] = SplitClientLayer(client, model_dir)

        print(self.model)
        self.model = self.model.eval()
        self.history = []
        self.count = 0

        self.in_queue = in_queue
        self.out_queue = out_queue

    def forward(self, x):
        if self.client is None:
            x = self.model.generate(x, max_length=512)
        else:
            x = self.tokenizer(x) # TODO need update
            x = self.model(**x)
        return x

    def backward(self):
        pass  # TODO need update

    def build_prompt(self) -> str:
        # prompt = "Welcome to the ChatGLM-6B model. Type your message."
        prompt = ""
        _, response = self.history[-1]
        prompt += response
        return prompt

    def process(self, query) -> str:
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

    def run(self):
        """
        Train the network on the training set.
        """

        model_dict = self.load_local()
        if model_dict:
            self.load_state_dict(model_dict["model_state_dict"])

        while True:
            while self.in_queue.empty():
                sleep(0.1)
            data = self.in_queue.get()

            if data["stage"] == "forward":
                self.forward(data)
            else: # stage == "backward"
                self.backward(data)


def main():
    chat_model = ChatModel()
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


if __name__ == "__main__":
    # main()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
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
