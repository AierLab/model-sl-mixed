from abc import ABC, abstractmethod


class AbstractServer(ABC):

    @abstractmethod
    def __init__(self):
        pass
