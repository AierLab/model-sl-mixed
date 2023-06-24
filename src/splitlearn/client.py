from comn import AbstractClient
from model import AbstractModel
from data import AbstractData
import torch


class Client(AbstractClient):
    def __init__(self, data: AbstractData, model: AbstractModel):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch_num = 1

        self.model = model
        self.trainloader, self.testloader = data.trainloader, data.testloader
        self.num_examples = data.get_number_examples()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))

    def fit(self, ckpt_path, config):
        self.set_parameters(ckpt_path)
        self.model.model_train(self.trainloader, self.epoch_num, self.device)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, ckpt_path, config):
        self.set_parameters(ckpt_path)
        loss, accuracy = self.model.model_test(self.testloader, self.device)
        print(f"loss: {loss}, accuracy{accuracy}")  # TODO refactor to log
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
        

