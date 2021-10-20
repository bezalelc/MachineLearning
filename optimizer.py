import numpy as np
import abc


class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self, data):
        pass


class Vanilla(Optimizer):

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * dW

    def save(self):
        return None

    def load(self, data):
        pass


class Momentum(Optimizer):

    def __init__(self, rho: float = 0.9) -> None:
        super().__init__()
        self.V: np.ndarray = np.array([0])
        self.rho: float = rho

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        V, rho = self.V, self.rho
        V = rho * V - alpha * dW
        self.V = V
        return np.array(-V)
        # self.V = rho * V + dW
        # return alpha * self.V

    def save(self):
        return self.V

    def load(self, data):
        self.V = data


class NesterovMomentum(Momentum):

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        V, rho = self.V, self.rho
        V_prev = V.copy()
        V = rho * V - alpha * dW
        self.V = V
        return rho * V_prev + (1 + rho) * V


class AdaGrad(Optimizer):

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.GradSquare: np.ndarray = np.array([0])
        self.eps: float = eps  # for numeric stability

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        GradSquare, eps = self.GradSquare, self.eps
        GradSquare += dW ** 2
        self.GradSquare = GradSquare
        return alpha * dW / (GradSquare ** 0.5 + eps)

    def save(self):
        return self.GradSquare

    def load(self, data):
        self.GradSquare = data


class RMSProp(AdaGrad):
    def __init__(self, eps: float = 1e-8, decay_rate: float = .999) -> None:
        super().__init__(eps)
        self.decay_rate: float = decay_rate

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        GradSquare, eps, decay_rate = self.GradSquare, self.eps, self.decay_rate
        GradSquare = decay_rate * GradSquare + (1 - decay_rate) * dW ** 2
        self.GradSquare = GradSquare
        return alpha * dW / (GradSquare ** 0.5 + eps)

    def save(self):
        return self.GradSquare, self.decay_rate

    def load(self, data):
        self.GradSquare, self.decay_rate = data


class Adam(Momentum, RMSProp):

    def __init__(self, rho: float = .9, decay_rate: float = .999, eps: float = 1e-8) -> None:
        Momentum.__init__(self, rho)
        RMSProp.__init__(self, eps, decay_rate)
        self.__t: int = 0
        self.rho_t: float = self.rho
        self.decay_rate_t: float = self.decay_rate

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        V, GradSquare, rho, eps, decay_rate = self.V, self.GradSquare, self.rho, self.eps, self.decay_rate
        self.rho_t, self.decay_rate_t = self.rho_t * self.rho, self.decay_rate_t * self.decay_rate
        rho_t, decay_rate_t = self.rho_t, self.decay_rate_t
        V = rho * V + (1 - rho) * dW
        GradSquare = decay_rate * GradSquare + (1 - decay_rate) * dW ** 2
        V_, GradSquare_ = V / (1 - rho_t), GradSquare / (1 - decay_rate_t)
        self.V, self.GradSquare = V, GradSquare
        return alpha * V_ / (GradSquare_ ** 0.5 + eps)

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, t: int):
        self.__t = t
        self.rho_t, self.decay_rate_t = self.rho ** t, self.decay_rate ** t

    def save(self):
        return self.GradSquare, self.V, self.rho_t, self.decay_rate_t

    def load(self, data):
        self.GradSquare, self.V, self.rho_t, self.decay_rate_t = data


class Newton(Optimizer):

    def __init__(self, X: np.ndarray) -> None:
        super().__init__()
        self.X_T: np.ndarray = X.T
        self.H_inv: np.ndarray = np.linalg.pinv(self.X_T @ X)

    def opt(self, dW: np.ndarray, alpha: float, y=0) -> np.ndarray:
        return self.H_inv @ self.X_T @ y

    def save(self):
        pass

    def load(self, data):
        pass
