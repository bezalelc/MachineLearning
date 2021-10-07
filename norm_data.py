import numpy as np
import abc
from typing import Union


class NormData(metaclass=abc.ABCMeta):

    def __init__(self, axis: Union[int, tuple] = 0, keep_dims: bool = True) -> None:
        super().__init__()
        self.axis: Union[int, tuple] = axis
        self.keep_dims: bool = keep_dims

    @abc.abstractmethod
    def norm(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def recover(self, X: np.ndarray) -> np.ndarray:
        pass


class StdNorm(NormData):
    """
    move the distribution of the data to mean=0 and std=1 (=normal distribution)
    """

    def __init__(self, axis: Union[int, tuple] = 0, keep_dims: bool = True) -> None:
        super().__init__(axis, keep_dims)
        self.mu: float = 0
        self.std: float = 0
        self.ddof: int = 0

    def norm(self, X: np.ndarray, X_train: bool = True) -> np.ndarray:
        if X_train:
            dims = self.keep_dims
            mu = np.mean(X, axis=self.axis, keepdims=dims)
            std = np.std(X, axis=self.axis, ddof=self.ddof, keepdims=dims)
            std[std == 0] = 1
            self.mu, self.std = mu, std
        else:
            mu, std = self.mu, self.std

        return (X - mu) / std

    def recover(self, X: np.ndarray) -> np.ndarray:
        mu, std = self.mu, self.std
        return X * std + mu


class ZeroCenter(NormData):
    """
    move the center of the data to 0
    """

    def __init__(self, axis: Union[int, tuple] = 0, keep_dims: bool = True) -> None:
        super().__init__(axis, keep_dims)
        self.mu: float = 0

    def norm(self, X: np.ndarray, X_train: bool = True) -> np.ndarray:
        if X_train:
            dims = self.keep_dims
            mu = np.mean(X, axis=self.axis, keepdims=dims)
            self.mu = mu
        else:
            mu = self.mu

        return X - mu

    def recover(self, X: np.ndarray) -> np.ndarray:
        return X + self.mu


class ZeroOneRange(NormData):
    """
    move data to range [0,1]
    """

    def __init__(self, axis: Union[int, tuple] = 0, keep_dims: bool = True) -> None:
        super().__init__(axis, keep_dims)
        self.min: float = 0
        self.max: float = 0
        self.div: float = 1

    def norm(self, X: np.ndarray, X_train: bool = True) -> np.ndarray:
        if X_train:
            axis, dims = self.axis, self.keep_dims
            min_, max_ = np.min(X, axis=axis, keepdims=dims), np.max(X, axis=axis, keepdims=dims)
            div = (max_ - min_)
            div[div == 0] = 1
            self.min, self.max, self.div = min_, max_, div
        else:
            min_, div = self.min, self.div

        return (X - min_) / div

    def recover(self, X: np.ndarray) -> np.ndarray:
        min_, div = self.min, self.div
        return X * div + min_


def standard_deviation(data, ddof=0):
    """
    Compute the standard deviation of data

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        mu: numpy array of the mean of every column (=attribute)
        sigma: numpy array of standard deviation of every column (=attribute)
    """
    '''
    m, n = data.shape
    mu = 1 / m * np.sum(data, axis=0)
    sigma = np.sqrt(1 / m * np.sum((data - mu) ** 2, axis=0))
    sigma[sigma == 0] = 1  // if sigma[i] == 0 => need to divide by 1 because there is not standard deviation  
    '''
    mu, sigma = np.mean(data, axis=0), np.std(data, axis=0, ddof=ddof)
    sigma[sigma == 0] = 1
    data = (data - mu) / sigma
    return data, mu, sigma


def simple_normalize(data):
    """
    Compute data between [-1,1]

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        max_: numpy array of max in every column (=attribute)
        min_: numpy array of min in every column (=attribute)
    """
    max_, min_ = np.max(data, axis=0), np.min(data, axis=0)
    div = (max_ - min_)
    div[div == 0] = 1
    data = (data - min_) / div
    return data, max_, min_


def sub_mean(data, axis: int = 0):
    return data - np.mean(data, axis=axis)
