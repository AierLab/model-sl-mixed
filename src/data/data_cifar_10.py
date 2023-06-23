import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .data_abstract import AbstractData


class CifarData(AbstractData):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        super().__init__()

    def _get_trainloader(self):
        """Load CIFAR-10 (training set)."""
        trainset = CIFAR10(self.data_dir, train=True, download=True, transform=self._transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

        return trainloader, len(trainset)

    def _get_testloader(self):
        """Load CIFAR-10 (test set)."""
        testset = CIFAR10(self.data_dir, train=False, download=True, transform=self._transform)
        testloader = DataLoader(testset, batch_size=32)

        return testloader, len(testset)
