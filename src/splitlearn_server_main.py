from comn import AbstractServer
import socket
import os
from splitlearn import SplitServer
import argparse
from model import DemoModel
from model import SplitServerModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ckpt",
        type=str,
        default="",
        help="Path to the checkpoint to be loaded",
    )

    parser.add_argument(
        "-host",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "-port",
        type=int,
        default=10086,
        help="Port of the server",
    )

    parser.add_argument(
        "-e",
        type=int,
        default=1,
        help="Number of epochs to train",
    )

    args = parser.parse_args()
    epoch_num = int(args.e)
    ckpt_path = args.ckpt
    host = args.host
    port = int(args.port)

    # init server
    SERVER_DIR = "../tmp/server"
    model = SplitServerModel(None, SERVER_DIR)

    model_dict = model.load_local()
    if model_dict:
        model.load_state_dict(model_dict["model_state_dict"])
        base_epoch = model_dict["epoch"]

    server = SplitServer(model, epoch_num, ckpt_path, host, port)

    server.run()
    server.fit()
