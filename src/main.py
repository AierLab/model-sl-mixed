import model
import data
import torch

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCH_NUM = 1
    CLIENT_DIR = "../tmp/client"

    # Init data and model.
    data = data.CifarData(data_dir=CLIENT_DIR)
    model = model.DemoModel(None, model_dir=CLIENT_DIR) # FIXME replace with an client object

    # Train model with trainloader.
    model.model_train(data.trainloader, EPOCH_NUM, DEVICE)

    # Test model with testloader.
    loss, accuracy = model.model_test(data.testloader, DEVICE)

    print(f"loss: {loss}, accuracy{accuracy}")