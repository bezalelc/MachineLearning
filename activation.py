import numpy as np
import abc


class Activation(metaclass=abc.ABCMeta):
    """
    interface for activation classes
    """

    @staticmethod
    @abc.abstractmethod
    def activation(X: np.ndarray) -> np.ndarray:
        """
        activation
        :param X: X@W+b
        :return: activation(X@W+b)
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def grad(H: np.ndarray) -> np.ndarray:
        """
        gradient for specific activation

        :param H: output of next layer

        :return: gradient
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        delta for chain rule if the activation is the last layer
        :param y: True classes
        :param pred: prediction classes
        :return: delta
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        """
        loss of activation if the activation is the last layer
        :param y: True classes
        :param pred: prediction classes
        :return: loss
        """
        pass


class Sigmoid(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-X))

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return H * (1 - H)

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        m, k = pred.shape[0], pred.shape[1]
        K = np.arange(k)
        delta = pred - np.array(y[:, None] == K)
        return delta

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m, k = pred.shape[:2]
        K = np.arange(k)
        pos = np.array(y == K[:, None]).T
        J = -(np.sum(np.log(pred[pos])) + np.sum(np.log(1 - pred[~pos]))) / m
        return J


class Relu(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class Linear(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class Softmax(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        X = np.exp(X)
        X /= np.sum(X, axis=1, keepdims=True)
        return X

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return H > 0

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        m = pred.shape[0]
        delta = pred.copy()
        delta[np.arange(m), y] -= 1
        return delta

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m = pred.shape[0]
        return float(np.sum(-np.log(pred[np.arange(m), y]))) / m


class Hinge(Activation):

    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return H

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        delta = Hinge.predict_matrix(y, pred)
        delta[delta > 0] = 1
        delta[np.arange(m), y] = - delta.sum(axis=1)
        return delta

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m = y.shape[0]
        return np.sum(Hinge.predict_matrix(y, pred)) / m
        # return np.sum(np.maximum(0, pred - pred[np.arange(m), y].reshape((-1, 1)) + 1)) / m - 1

    @staticmethod
    def predict_matrix(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = pred.copy()
        P = P - P[np.arange(m), y].reshape((-1, 1)) + 1
        P[P < 0], P[np.arange(m), y] = 0, 0
        return P
