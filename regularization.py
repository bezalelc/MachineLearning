import numpy as np
import abc


class Regularization(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def norm(x: np.ndarray, axis: int = None) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        pass


class L1(Regularization):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None) -> np.ndarray:
        return np.sum(np.abs(x), axis=axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return np.abs(x)


class L2(Regularization):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None, sqrt_=False) -> np.ndarray:
        if sqrt_:
            return np.sum(x ** 2, axis=axis) ** 0.5
        return np.sum(x ** 2, axis=axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return x * 2


class L12(Regularization):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None) -> float:
        return L1.norm(x, axis) + L2.norm(x, axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return L1.d_norm(x) + L2.d_norm(x)
