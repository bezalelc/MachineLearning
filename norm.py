import numpy as np
import abc


class Norm(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def norm(x: np.ndarray, axis: int = None) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        pass


class Norm1(Norm):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None) -> np.ndarray:
        return np.sum(np.abs(x), axis=axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return np.abs(x)


class Norm2(Norm):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None, sqrt_=False) -> np.ndarray:
        if sqrt_:
            return np.sum(x ** 2, axis=axis) ** 0.5
        return np.sum(x ** 2, axis=axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return x * 2


class Norm12(Norm):

    @staticmethod
    def norm(x: np.ndarray, axis: int = None) -> float:
        return Norm1.norm(x, axis) + Norm2.norm(x, axis)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return Norm1.d_norm(x) + Norm2.d_norm(x)
