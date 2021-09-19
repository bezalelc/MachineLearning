import abc
import numpy as np
from numpy import random
from typing import Union
from activation import Activation_, Hinge_, Softmax_, Sigmoid, Activation
from regularization import Regularization, L1, L2, L12
from optimizer import Optimizer, Vanilla


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, out_shape: int, input_shape: tuple = None) -> None:
        super().__init__()
        self.__input_shape: tuple = input_shape
        self.out_shape: int = out_shape

        # temp
        self.W = None
        self.alpha, self.lambda_ = 0, 0
        self.bias: bool = True
        self.reg = None
        self.act = None
        self.numeric_stability = True

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        pass

    def grad(self, X: np.ndarray, h) -> None:
        pass

    @abc.abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def regularize(self) -> np.ndarray:
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        self.__input_shape = input_shape_


class Dense(Layer):
    def __init__(self, out_shape: int, activation: Activation = Sigmoid, reg: Regularization = L2,
                 opt: Optimizer = Vanilla, eps=1e-3, alpha=1e-5, input_shape: tuple = None,
                 lambda_=0, add_bais=True, reg_bias=False, opt_bias=True) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.eps: float = eps
        self.add_bias: bool = add_bais
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.W = None
        self.__input_shape = input_shape

        if input_shape:
            assert len(input_shape) == 1
            self.W = random.uniform(low=-self.eps, high=self.eps, size=(input_shape[0] + add_bais, self.out_shape))

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = activation
        self.reg: Regularization = reg
        self.opt: Optimizer = opt

    def forward(self, X: np.ndarray) -> np.ndarray:
        Act, W = self.act.activation, self.W
        if self.add_bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return Act(X @ W)

    def backward(self, delta: np.ndarray, h: np.ndarray = None) -> None:
        pass
        # Opt, W, alpha = self.opt.opt, self.W, self.alpha
        # W -= Opt(self.grad(X, h), alpha)

    def grad(self, X: np.ndarray, h):
        pass
        # Grad, W, lambda_, Reg, dReg = self.act.grad, self.W, self.lambda_, self.reg.norm, self.reg.d_norm
        # return Grad(X, W, h) + lambda_ * dReg(W)

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: list):
        self.__input_shape = input_shape_
        add_bias = self.add_bias
        if input_shape_:
            assert len(input_shape_) == 1
            self.W = random.uniform(low=-self.eps, high=self.eps, size=(input_shape_[0] + add_bias, self.out_shape))


class Dense1(Layer):
    def __init__(self, out_shape: int, activation: Activation = Sigmoid, reg: Regularization = L2,
                 opt: Optimizer = Vanilla, eps=1e-3, alpha=1e-5, input_shape: tuple = None,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.m = 1
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.W = None
        self.b = random.uniform(low=-self.eps, high=self.eps, size=(out_shape,)) if bias else None
        self.__input_shape = input_shape

        if input_shape:
            assert len(input_shape) == 1
            self.W = random.uniform(low=-self.eps, high=self.eps, size=(input_shape[0], self.out_shape))
            # TODO remove next 3 line
            np.random.seed(0)
            self.W = random.randn(input_shape[0] + 1, self.out_shape) * eps
            self.W, self.b = self.W[1:, :], self.W[0, :]

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = activation
        self.reg: Regularization = reg
        self.opt: Optimizer = opt

    def forward(self, X: np.ndarray) -> np.ndarray:
        Act, W, b = self.act.activation, self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.numeric_stability:
            Z -= np.max(Z, axis=1, keepdims=True)
        H = Act(Z)

        return H

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt.opt
        W, b, alpha, lambda_, m, bias = self.W, self.b, self.alpha, self.lambda_, h.shape[0], self.bias
        # dW = (h.T @ delta + lambda_ * dReg(W) / 2) / m
        dW = h.T @ delta / m + lambda_ * dReg(W)

        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b)
                # db /= m
            b -= Opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = self.grad(delta, h)

        # TODO remove next line
        # print(np.sum(dW), np.sum(db),np.sum(dW)+ np.sum(db),)

        W -= Opt(dW, alpha)
        return delta

    def grad(self, delta: np.ndarray, h: np.ndarray) -> np.ndarray:
        dAct, W = self.act.grad, self.W
        return delta @ W.T * dAct(h)

    def regularize(self) -> Union[int, np.ndarray]:
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
        bias = self.bias
        if input_shape_:
            assert len(input_shape_) == 1
            self.W = random.uniform(low=-self.eps, high=self.eps, size=(input_shape_[0], self.out_shape))
            if bias:
                self.b = random.uniform(low=-self.eps, high=self.eps, size=(self.out_shape,))

    def __str__(self) -> str:
        s = f'W.shape: ' + str(self.W.shape)
        return s
