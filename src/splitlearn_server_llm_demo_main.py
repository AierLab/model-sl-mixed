from queue import Queue
from threading import Thread

from model import ChatModel
from splitlearn import SplitServer

if __name__ == "__main__":
    SERVER_DIR = "../tmp/server"

    in_queue = Queue()
    out_queue = Queue()

    # Init data and model.
    model = ChatModel(SERVER_DIR, in_queue, out_queue)
    server = SplitServer(in_queue, out_queue)

    t1 = Thread(target=model.run)
    t2 = Thread(target=server.run, args=("localhost", 8888))

    t1.start()
    t2.start()
