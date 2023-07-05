from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel


class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        self.history = []

    def build_prompt(self) -> str:
        # prompt = "Welcome to the ChatGLM-6B model. Type your message."
        prompt = ""
        _, response = self.history[-1]
        prompt += response
        return prompt

    def process(self, query: str) -> str:
        count = 0
        for response, self.history in self.model.stream_chat(self.tokenizer, query, history=self.history):
            count += 1
            # if count % 8 == 0:
            #     yield self.build_prompt()
        return self.build_prompt()

    def clear(self) -> None:
        self.history = []

    def stop(self) -> None:
        self.model = None


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
    main()
