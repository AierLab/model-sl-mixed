from data import CifarData
from fedlearn import FedClient
from model import DemoModel

if __name__ == '__main__':

    # run in separate terminal
    # CLIENT_DIR = "../tmp/client/c01"
    CLIENT_DIR = "../tmp/client/c02"
    # CLIENT_DIR = "../tmp/client/c01"

    # Init data and model.
    data = CifarData(data_dir=CLIENT_DIR)
    model = DemoModel(None, model_dir=CLIENT_DIR)

    client = FedClient(data, model)
    client.run()