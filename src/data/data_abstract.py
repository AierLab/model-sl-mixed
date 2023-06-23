from abc import ABC, abstractmethod


class AbstractData(ABC):
    def __init__(self):
        self.trainloader, self._train_length = self._get_trainloader()
        self.testloader, self._test_length = self._get_testloader()

    def get_number_examples(self):
        num_examples = {"trainset": self._train_length, "testset": self._test_length}  # TODO refactor: key rename
        return num_examples

    @abstractmethod
    def _get_trainloader(self):
        pass

    @abstractmethod
    def _get_testloader(self):
        pass
