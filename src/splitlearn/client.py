from data import AbstractData
import torch
import helper
from model import SplitClientModel


class SplitClient:
    def __init__(self, data: AbstractData, model: SplitClientModel):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch_num = 1

        self.model = model
        self.trainloader, self.testloader = data.trainloader, data.testloader
        self.num_examples = data.get_number_examples()

    def fit(self):
        self.model.model_train(self.trainloader, self.epoch_num, self.device)
        return helper.get_weights(self.model), self.num_examples["trainset"], {}

    def evaluate(self):
        loss, accuracy = self.model.model_test(self.testloader, self.device)
        print(f"loss: {loss}, accuracy{accuracy}")  # TODO refactor to log
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def run(self):
        print(self.fit())
