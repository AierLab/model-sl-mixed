
import argparse
import flwr as fl

from comn import AbstractServer
from model import AbstractModel, DemoModel
from helper import get_weights, set_weights


class Server(AbstractServer):
    def __init__(self, model: AbstractModel, fc, ac):
        # get weights of model
        init_weights = get_weights(model)
        # Convert the weights (np.ndarray) to parameters
        init_param = fl.common.ndarrays_to_parameters(init_weights)

        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            # fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=fc,  # Never sample less than 2 clients for training
            # min_evaluate_clients=ec,  # Never sample less than 2 clients for evaluation
            min_available_clients=ac,  # Wait until all 2 clients are available
            initial_parameters=init_param,
        )

    def run(self):
        fl.server.start_server(server_address="localhost:8080",
                               config=fl.server.ServerConfig(num_rounds=3),
                               strategy=self.strategy)


if __name__ == "__main__":
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

    server = Server(model, fc, ac)
    server.run()
