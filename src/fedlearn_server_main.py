import argparse

from model import DemoModel
from fedlearn import FedServer

if __name__ == '__main__':
    # Start Flower server for three rounds of federated learning
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )

    parser.add_argument("-b", type=int, default=32, help="Batch size")

    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )

    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )

    parser.add_argument(
        "-ec",
        type=int,
        default=2,
        help="Min evaluate clients, min number of clients to be sampled for evaluation",
    )

    parser.add_argument(
        "-ckpt",
        type=str,
        default="",
        help="Path to checkpoint to be loaded",
    )

    args = parser.parse_args()

    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    ec = int(args.ec)
    ckpt_path = args.ckpt
    global batch_size
    batch_size = int(args.b)
    init_param = None

    SERVER_DIR = "../tmp/server"
    model = DemoModel(None, SERVER_DIR)
    base_epoch = 0

    model_dict = model.load_local()
    if model_dict:
        model.load_state_dict(model_dict["model_state_dict"])
        base_epoch = model_dict["epoch"]

    server = FedServer(model, base_epoch, fc, ac)
    server.run()
