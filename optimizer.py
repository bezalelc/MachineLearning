import numpy as np
import abc


class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        pass


class Vanilla(Optimizer):

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * dW


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


class RMSProp(AdaGrad):
    def __init__(self, eps: float = 1e-8, decay_rate: float = .999) -> None:
        super().__init__(eps)
        self.decay_rate: float = decay_rate

    def opt(self, dW: np.ndarray, alpha: float) -> np.ndarray:
        GradSquare, eps, decay_rate = self.GradSquare, self.eps, self.decay_rate
        GradSquare = decay_rate * GradSquare + (1 - decay_rate) * dW ** 2
        self.GradSquare = GradSquare
        return alpha * dW / (GradSquare ** 0.5 + eps)


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

    # def f(self):
    #     pass
