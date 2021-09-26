import numpy as np
import abc


class Activation_(metaclass=abc.ABCMeta):
    """
    interface for activation classes
    """

    @staticmethod
    @abc.abstractmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        pass


class Hinge_(Activation_):
    """
    hinge activation for svm
    """

    @staticmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return X @ W

    @staticmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = Hinge_.predict_matrix(X, W, y)
        P[P > 0] = 1
        P[np.arange(m), y] = - P.sum(axis=1)
        dW = X.T @ P / m
        return dW

    @staticmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        P = Hinge_.activation(X, W)
        return np.sum(np.maximum(0, P - P[np.arange(m), y].reshape((-1, 1)) + 1)) / m - 1

    @staticmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m = y.shape[0]
        P = Hinge_.predict_matrix(X, W, y)
        L = np.sum(P) / m

        # grad
        P[P > 0] = 1
        P[np.arange(m), y] = - P.sum(axis=1)
        dW = X.T @ P / m

        return L, dW

    @staticmethod
    def predict_matrix(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = Hinge_.activation(X, W)
        P = P - P[np.arange(m), y].reshape((-1, 1)) + 1
        P[P < 0], P[np.arange(m), y] = 0, 0
        return P

    @staticmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m, n, k = X.shape[0], X.shape[1], W.shape[1]
        P = Hinge_.predict_matrix(X, W, y)

        L, dW = 0, np.zeros(W.shape)
        for i in range(m):
            for j in range(k):
                if j != y[i] and P[i, j] > 0:
                    dW[:, y[i]] -= X[i]
                    dW[:, j] += X[i]
                    L += P[i, j]
                    print(X[i, 0])

        L, dW = L / m, dW / m
        return L, dW


class Softmax_(Activation_):

    @staticmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        P = X @ W
        P -= np.max(P, axis=1)[..., None]
        P = np.exp(P)
        P /= np.sum(P, axis=1)[..., None]
        return P

    @staticmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = Softmax_.activation(X, W)
        P[np.arange(m), y] -= 1
        dW = X.T @ P / m

        return dW

    @staticmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        P = Softmax_.activation(X, W)
        L = np.sum(-np.log(P[np.arange(m), y])) / m
        return float(L)

    @staticmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m = y.shape[0]
        P = Softmax_.activation(X, W)

        # loss
        L = np.sum(-np.log(P[np.arange(m), y])) / m

        # grad
        P[np.arange(m), y] -= 1
        dW = X.T @ P / m

        return L, dW

    @staticmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        calculate the loss and gradient iin loop

        :param X:
        :param W:
        :param y:

        :return:
        """
        m, n, k = X.shape[0], X.shape[1], W.shape[1]

        P = Softmax_.activation(X, W)

        L, dW = 0, np.zeros(W.shape)
        for i in range(m):
            L += -np.log(P[i, y[i]])
            dW[:, y[i]] -= X[i]

            for j in range(k):
                dW[:, j] += X[i] * P[i, j]

        L, dW = L / m, dW / m
        return L, dW


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
        return delta / m

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m, k = pred.shape[:2]
        K = np.arange(k)
        pos = np.array(y == K[:, None]).T
        J = -(np.sum(np.log(pred[pos])) + np.sum(np.log(1 - pred[~pos]))) / m
        return J


class Linear(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return np.asarray(1.)

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
        return delta / m

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m = pred.shape[0]
        # pred[np.arange(m), y] += 1e-310

        return float(np.sum(-np.log(pred[np.arange(m), y]))) / m


class SoftmaxStable(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        Z = np.sum(np.exp(X), axis=1, keepdims=True)
        Z = X - np.log(Z)
        return Z

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return H > 0

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        m = pred.shape[0]
        delta = pred.copy()
        delta[np.arange(m), y] -= 1
        return delta / m

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        m = pred.shape[0]
        return -np.sum(pred[np.arange(m), y]) / m


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
        return delta / m

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


class Tanh(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class ReLU(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        return H > 0

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        return 0


class LeakyReLU(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        return np.maximum(0.01 * X, X)

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class PReLU(Activation):

    @staticmethod
    def activation(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.maximum(alpha * X, X)

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class ELU(Activation):
    @staticmethod
    def activation(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        A = X.copy()
        A[A < 0] = alpha * (np.exp(X) - 1)
        return A

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass


class MaxOut(Activation):
    @staticmethod
    def activation(X: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def grad(H: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def delta(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def loss(y: np.ndarray, pred: np.ndarray) -> float:
        pass
