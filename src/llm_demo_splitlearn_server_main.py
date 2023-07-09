import pickle
from collections import OrderedDict
from queue import Queue
from threading import Thread
from time import sleep

import torch.nn
from transformers import AutoTokenizer, AutoModel
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from model import SplitServerModel
from model.chatglm_6b_split_server import ChatGLMForConditionalGeneration, ChatGLMTokenizer, ChatGLMConfig
from splitlearn import SplitClient, SplitServer


def main():
    SERVER_DIR = "../tmp/server"

    in_queue = Queue()
    out_queue = Queue()

    # Init data and model.
    model = SplitServerModel(SERVER_DIR, in_queue, out_queue)
    server = SplitServer(in_queue, out_queue)

    def run():
        while True:
            response = model.process()
            print(response)

    t1 = Thread(target=run)
    t2 = Thread(target=server.run, args=("localhost", 8888))

    t1.start()
    t2.start()

if __name__ == "__main__":
    main()
