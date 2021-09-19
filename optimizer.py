import numpy as np
import abc


class Optimizer(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def opt(dW, alpha) -> np.ndarray:
        pass


class Vanilla(Optimizer):

    @staticmethod
    def opt(dW, alpha) -> np.ndarray:
        return alpha * dW
