
import argparse
import flwr as fl
from model import MyModel
from helpers import get_weights, set_weights

def main():
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

    if ckpt_path != "":
        model = MyModel()
        init_weights = get_weights(model)
        # Convert the weights (np.ndarray) to parameters
        init_param = fl.common.weights_to_parameters(init_weights)
        # del the net as we don't need it anymore
        del model

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        #fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=fc,  # Never sample less than 2 clients for training
        #min_evaluate_clients=ec,  # Never sample less than 2 clients for evaluation
        min_available_clients=ac,  # Wait until all 2 clients are available
        initial_parameters=init_param,
    )

    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": rounds}, strategy=strategy)
    
if __name__ == "__main__":
    main()