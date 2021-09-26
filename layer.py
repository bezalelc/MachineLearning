import abc
import numpy as np
from numpy import random
from typing import Union
from activation import Sigmoid, Activation, ReLU, Softmax, Linear, PReLU, Hinge, ELU, Tanh, SoftmaxStable
from regularization import Regularization, L2, L1, L12
from optimizer import Optimizer, Vanilla, Adam, AdaGrad, RMSProp, Momentum, NesterovMomentum
from weights_init import InitWeights, stdScale


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, out_shape: int, input_shape: tuple = None, ns: bool = False) -> None:
        super().__init__()
        self.__input_shape: tuple = input_shape
        self.out_shape: int = out_shape
        self.ns = ns

    @abc.abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        pass

    @abc.abstractmethod
    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param pred: input of the current layer

        :return: delta of the prediction
        """
        pass

    @abc.abstractmethod
    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """
        loss of linear layer if the layer is the last layer

        :param y: True classes
        :param pred: input of the current layer

        :return: loss of the prediction
        """
        pass

    @abc.abstractmethod
    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        """
        dict of grades for current layer if the layer is the last layer

        :param delta: True classes
        :param X: input of the current layer

        :return: loss of the prediction
        """
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        self.__input_shape = input_shape_


class ActLayer(Layer):

    def __init__(self, out_shape: int, input_shape: tuple = None, ns: bool = False, act: Activation = ReLU()) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.act: Activation = act

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        if self.ns:
            Z -= np.max(Z, axis=1, keepdims=True)
        return self.act.activation(Z)

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        return self.grad(delta, X)['delta']

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        dAct = self.act.grad
        return {'delta': delta * dAct(X)}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return self.act.delta(y, pred)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred)


class WeightLayer(Layer):
    """
    layer with weights and activation
    """

    def __init__(self, out_shape: int, reg: Regularization = L2, ns: bool = False,
                 opt: Optimizer = None, opt_bias_: Optimizer = None, eps=1e-3, alpha=1e-5,
                 input_shape: tuple = None,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W=stdScale, seed=-1) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.seed = seed

        # general params
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W = None
        # TODO remove
        # np.random.seed(6)
        self.b = init_W((out_shape,), eps) if bias else None
        self.__input_shape = input_shape
        if input_shape:
            assert len(input_shape) == 1
            # TODO remove
            if seed >= 0:
                np.random.seed(seed)
            self.W = init_W((input_shape[0], out_shape), eps)

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.reg: Regularization = reg
        self.opt: Optimizer() = opt if opt else Adam()
        self.opt_bias_: Optimizer() = opt_bias_ if opt_bias_ or not bias else Adam()

    def forward(self, X: np.ndarray) -> np.ndarray:
        W, b = self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.ns:
            Z -= np.max(Z, axis=1, keepdims=True)

        return Z

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        W, b, alpha, bias, Opt = self.W, self.b, self.alpha, self.bias, self.opt

        grades = self.grad(delta, X)
        dW = grades['dW']
        W -= Opt.opt(dW, alpha)
        if bias:
            db = grades['db']
            b -= self.opt_bias_.opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = grades['delta']

        return delta

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        """
        gradient for specific activation

        :param delta: gradient of the next layer
        :param X: input of the current layer

        :return: gradient of the current layer for the backward
        """
        dReg, Opt = self.reg.d_norm, self.opt
        W, b, alpha, lambda_, m, bias = self.W, self.b, self.alpha, self.lambda_, X.shape[0], self.bias
        grades = {'delta': delta @ W.T, 'dW': X.T @ delta + lambda_ * dReg(W) / 2}

        if bias:
            db = delta.sum(axis=0)
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            grades['db'] = db

        return grades

    def delta(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param h: input of the current layer

        :return: delta of the prediction
        """
        return Linear.delta(y, h)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """
         loss of linear layer if the layer is the last layer

         :param y: True classes
         :param pred: input of the current layer

         :return: loss of the prediction
         """
        return Linear.loss(y, pred)

    def regularize(self) -> float:
        if not self.reg:
            return 0

        Reg, W, b, lambda_ = self.reg.norm, self.W, self.b, self.lambda_
        r = Reg(W) * lambda_
        if self.reg_bias:
            r += Reg(b) * lambda_

        return r

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: list):
        self.__input_shape = input_shape_
        bias, eps = self.bias, self.eps
        if input_shape_:
            assert len(input_shape_) == 1
            # TODO remove
            if self.seed >= 0:
                np.random.seed(self.seed)
            self.W = self.init_W((input_shape_[0], self.out_shape), eps)
            if bias:
                self.b = self.init_W((self.out_shape,), eps)


class NormLayer(Layer):

    def __init__(self, out_shape: int, input_shape: tuple = None, eps=1e-5, axis=0, momentum=.9,
                 beta_opt: Optimizer = None, gamma_opt: Optimizer = None, alpha: float = 1e-7) -> None:
        super().__init__(out_shape, input_shape, False)
        self.axis: int = axis
        self.gamma: np.ndarray = np.ones(self.out_shape, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(self.out_shape, dtype=np.float64)
        self.running_mu: np.ndarray = np.zeros(self.out_shape, dtype=np.float64)
        self.running_var: np.ndarray = np.zeros(self.out_shape, dtype=np.float64)
        if axis == 1:
            self.gamma = self.gamma.reshape((-1, 1))
            self.beta = self.beta.reshape((-1, 1))
            self.running_mu = self.running_mu.reshape((-1, 1))
            self.running_var = self.running_var.reshape((-1, 1))

        self.mu = 0
        self.var = 0
        self.std = 0
        self.z = 0
        self.eps = eps
        self.alpha = alpha
        self.momentum = momentum
        self.beta_opt: Optimizer() = beta_opt if beta_opt else Adam()
        self.gamma_opt: Optimizer() = gamma_opt if gamma_opt else Adam()

    def forward(self, X: np.ndarray, mode='train') -> np.ndarray:
        gamma, beta, layer_norm, eps, axis = self.gamma, self.beta, self.axis, self.eps, self.axis
        if mode == 'train':
            # mu, var = X.mean(axis=axis), X.var(axis=axis)
            mu, var = X.mean(axis=axis), X.var(axis=axis) + eps
            std = var ** 0.5

            if axis == 1:
                X = X.T.copy()

            z = (X - mu) / std
            # z = (X - mu) / (std + eps)
            out = gamma * z + beta

            if axis == 0:
                momentum = self.momentum
                self.running_mu = momentum * self.running_mu + (1 - momentum) * mu
                self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.std, self.var, self.mu, self.z = std, var, mu, z

        elif mode == 'test':
            out = gamma * (X - self.running_mu) / (self.running_var + eps) ** 0.5 + beta


        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)

        if axis == 1:
            out = out.T
        return out

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> Union[np.ndarray, tuple]:
        grades, alpha = self.grad(delta, X), self.alpha
        d_beta, d_gamma, delta = grades['d_beta'], grades['d_gamma'], grades['delta']

        self.beta -= self.beta_opt.opt(d_beta, alpha)
        self.gamma -= self.gamma_opt.opt(d_gamma, alpha)

        return delta

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        z, gamma, beta, mu, var, std, axis = self.z, self.gamma, self.beta, self.mu, self.var, self.std, self.axis
        if axis == 1:
            delta = delta.T

        d_beta, d_gamma = delta.sum(axis=axis), np.sum(delta * z, axis=axis)

        m = delta.shape[0]
        df_dz = delta * gamma
        delta = (1 / (m * std)) * (m * df_dz - np.sum(df_dz, axis=0) - z * np.sum(df_dz * z, axis=0))
        if axis == 1:
            delta, d_beta, d_gamma = delta.T, d_beta.reshape(beta.shape), d_gamma.reshape(gamma.shape)

        grades = {'delta': delta, 'd_beta': d_beta, 'd_gamma': d_gamma}

        return grades

    # @property
    # def input_shape(self) -> tuple:
    #     return self.__input_shape

    # @input_shape.setter
    # def input_shape(self, input_shape_: tuple):
    #     self.__input_shape = input_shape_
    #     if input_shape_:
    #         assert len(input_shape_) == 1
    #         self.init_array()
    #
    # def init_array(self):
    #     if self.axis == 1:
    #         self.gamma = self.gamma.reshape((-1, 1))
    #         self.beta = self.beta.reshape((-1, 1))
    #         # self.gamma = self.gamma.reshape((-1, 1))
    #         # self.gamma = self.gamma.reshape((-1, 1))


class Dropout(Layer):

    def __init__(self, out_shape: int, input_shape: tuple = None, ns: bool = False, p: float = 0.5, seed=-1) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.p: float = p
        self.mask: np.ndarray = np.array([])
        self.seed = seed

    def forward(self, X: np.ndarray, mode: str = 'train') -> np.ndarray:
        if self.seed >= 0:
            np.random.seed(self.seed)

        if mode == 'train':
            p, X = self.p, X.copy()
            self.mask = (np.random.rand(*X.shape) < p) / p
            X = X * self.mask

        return X

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True, mode: str = 'train') -> np.ndarray:
        return self.grad(delta, X, mode)['delta']

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def grad(self, delta: np.ndarray, X: np.ndarray, mode: str = 'train') -> dict[str:np.ndarray]:
        mask = self.mask
        return {'delta': delta * mask if mode == 'train' else delta}


class SoftmaxStableLayer(ActLayer):

    def __init__(self, out_shape: int, input_shape: tuple = None, ns: bool = True) -> None:
        super().__init__(out_shape, input_shape, ns, SoftmaxStable())
        self.log_prob = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        if self.ns:
            Z -= np.max(Z, axis=1, keepdims=True)
        self.log_prob = self.act.activation(Z)
        return np.exp(self.log_prob)

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        return super().grad(delta, X)

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        return super().backward(delta, X, return_delta)

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return super().delta(y, pred)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return super().loss(y, self.log_prob)


class Conv(Layer):

    def __init__(self, out_shape: int, input_shape: tuple = None, ns: bool = False, stride: int = 1,
                 pad: float = 0) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.stride: int = stride
        self.pad: float = pad

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        pass

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        pass


# ****************************   Complex Layers   *************************
class Dense(Layer):
    """
    layer with weights and activation
    """

    def __init__(self, out_shape: int, act: Activation = Sigmoid, reg: Regularization = L2,
                 opt: Optimizer = Vanilla(), opt_bias_: Optimizer = Vanilla(), eps=1e-3, alpha=1e-5,
                 input_shape: tuple = None,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W=stdScale) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.m = 1
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W = None
        self.b = init_W((out_shape,), eps) if bias else None
        self.__input_shape = input_shape

        if input_shape:
            assert len(input_shape) == 1
            self.W = init_W((input_shape[0], out_shape), eps)

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = act
        self.reg: Regularization = reg
        self.opt: Optimizer() = opt
        self.opt_bias_: Optimizer() = opt_bias_

        if isinstance(self.act, ReLU):
            self.numeric_stability = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        Act, W, b = self.act.activation, self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.numeric_stability:
            Z -= np.max(Z, axis=1, keepdims=True)
        self.Z = Z
        H = Act(Z)

        return H

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, X.shape[0], self.bias
        delta = delta * dAct(Z)  # activation grad

        dW = X.T @ delta / m + lambda_ * dReg(W) / 2
        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            # print(db.shape)
            # print(Opt.opt(db, alpha).shape)
            # b -= Opt.opt(db, alpha)
            b -= self.opt_bias_.opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = delta @ W.T

        # TODO remove next 3 line
        # print(dW*m, db*m, np.sum(dW) + np.sum(db))
        # print(np.sum(np.abs(dW)), np.sum(np.abs(db)), np.sum(dW) + np.sum(db))
        # print(np.su)

        W -= Opt.opt(dW, alpha)
        return delta

    def grad(self, delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        gradient for specific activation

        :param delta: gradient of the next layer
        :param pred: input of the current layer

        :return: gradient of the current layer for the backward
        """
        dAct, W, Z = self.act.grad, self.W, self.Z
        return delta @ W.T * dAct(Z)

    def delta(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param h: input of the current layer

        :return: delta of the prediction
        """
        return self.act.delta(y, h)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred)

    def regularize(self) -> float:
        if not self.reg:
            return 0

        Reg, W, b, lambda_ = self.reg.norm, self.W, self.b, self.lambda_
        r = Reg(W) * lambda_
        if self.reg_bias:
            r += Reg(b) * lambda_

        return r

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: list):
        self.__input_shape = input_shape_
        bias, eps = self.bias, self.eps
        if input_shape_:
            assert len(input_shape_) == 1
            self.W = self.init_W((input_shape_[0], self.out_shape), eps)
            # self.W = random.randn(input_shape_[0], self.out_shape) * eps
            if bias:
                self.b = self.init_W((self.out_shape,), eps)
                # self.b = random.randn(self.out_shape) * eps

    def __str__(self) -> str:
        s = f'W.shape: ' + str(self.W.shape)
        return s


class ComplexLayer(Layer):

    def __init__(self, out_shape: int, act: Activation = ReLU, norm: NormLayer = None, reg: Regularization = L2,
                 opt: Optimizer = Adam(), opt_bias_: Optimizer = Vanilla(), eps=1e-3, alpha=1e-5,
                 input_shape: tuple = None, lambda_=0, bias=True, reg_bias=False, opt_bias=True,
                 init_W=stdScale) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W = None
        self.b = init_W((out_shape,), eps) if bias else None
        self.__input_shape = input_shape

        if input_shape:
            assert len(input_shape) == 1
            self.W = init_W((input_shape[0], out_shape), eps)

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = act
        self.reg: Regularization = reg
        self.norm: NormLayer = norm
        self.opt: Optimizer() = opt
        self.opt_bias_: Optimizer() = opt_bias_

        if isinstance(self.act, ReLU):
            self.numeric_stability = False

    def forward(self, X: np.ndarray, mode: str = 'train') -> np.ndarray:
        Act, W, b = self.act.activation, self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.numeric_stability:
            Z -= np.max(Z, axis=1, keepdims=True)

        self.Z = Z
        if self.norm:
            Z = self.norm.forward(Z, mode)
        H = Act(Z)

        return H

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, X.shape[0], self.bias
        delta = delta * dAct(Z)  # activation grad

        dW = X.T @ delta / m + lambda_ * dReg(W) / 2
        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            # print(db.shape)
            # print(Opt.opt(db, alpha).shape)
            # b -= Opt.opt(db, alpha)
            b -= self.opt_bias_.opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = delta @ W.T

        # TODO remove next 3 line
        # print(dW*m, db*m, np.sum(dW) + np.sum(db))
        # print(np.sum(np.abs(dW)), np.sum(np.abs(db)), np.sum(dW) + np.sum(db))
        # print(np.su)

        W -= Opt.opt(dW, alpha)
        return delta

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return self.act.delta(y, pred)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred)

    def grad(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> dict:
        dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, X.shape[0], self.bias
        grades = {}
        delta = delta * dAct(Z)  # activation grad

        grades['dW'] = X.T @ delta / m + lambda_ * dReg(W) / 2

        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
                grades['db'] = db

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            grades['delta'] = delta @ W.T

        return grades
