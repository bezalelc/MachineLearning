import abc
import numpy as np
from numpy import random
from typing import Union
from activation import Sigmoid, Activation, ReLU, Linear, SoftmaxStable, Tanh
from norm import Norm, Norm2
from optimizer import Optimizer, Adam
from weights_init import InitWeights, xavierScale, stdScale


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None,
                 ns: bool = False) -> None:
        super().__init__()
        self.__input_shape: Union[int, tuple] = input_shape
        self.out_shape: Union[int, tuple] = out_shape
        self.ns: bool = ns

    @abc.abstractmethod
    def forward(self, X: Union[np.ndarray, tuple]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, delta: np.ndarray, X: Union[np.ndarray, tuple], return_delta=True) -> np.ndarray:
        pass

    @abc.abstractmethod
    def grad(self, delta: np.ndarray, X: Union[np.ndarray, tuple]) -> dict[str:np.ndarray]:
        """
        dict of grades for current layer if the layer is the last layer

        :param delta: True classes
        :param X: input of the current layer

        :return: loss of the prediction
        """
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

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: Union[int, tuple]) -> None:
        pass

    def save(self):
        pass

    def load(self):
        pass

    @abc.abstractmethod
    def reshape(self, data, in_=True):
        pass

    # @abc.abstractmethod
    # def init(self):
    #     pass


class ActLayer(Layer):

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None, ns: bool = False,
                 act: Activation = ReLU(), data: dict = None) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.__input_shape: Union[int, tuple] = input_shape
        self.act: Activation = act
        self.data: dict = data if data else {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = self.reshape(X)

        Z = X.copy()
        if self.ns:
            Z -= np.max(Z, axis=-1, keepdims=True)
        return self.act.activation(Z)

    def backward(self, delta: np.ndarray, X: np.ndarray, H: np.ndarray = None, return_delta=True) -> np.ndarray:
        return self.grad(delta, X, H)['delta']

    def grad(self, delta: np.ndarray, X: np.ndarray, H: np.ndarray = None) -> dict[str:np.ndarray]:
        dAct = self.act.grad
        X, delta = self.reshape(X, True), self.reshape(delta, True)
        if H is None:
            H = self.act.activation(X)

        return {'delta': delta * dAct(H)}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return self.act.delta(y, pred, self.data)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred, self.data)

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        self.__input_shape = input_shape_
        if not self.out_shape:
            self.out_shape = input_shape_

    def reshape(self, data, in_=True):
        if len(data.shape) > 2:
            self.data['m'] = data.shape[0]
        return np.reshape(data, (-1,) + (self.input_shape if in_ else self.out_shape))
        # return np.reshape(data, (-1,) + ((self.input_shape[-1],) if in_ else self.out_shape))


class WeightLayer(Layer):
    """
    layer with weights and activation
    """

    def __init__(self, out_shape: Union[int, tuple], input_shape: Union[int, tuple] = 0, reg: Norm = Norm2,
                 ns: bool = False, opt: Optimizer = None, opt_bias_: Optimizer = None, eps=1e-3, alpha=1e-5,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W: InitWeights = xavierScale,
                 seed=-1) -> None:
        input_shape = (int(np.prod(input_shape)),) if input_shape else None
        super().__init__((int(out_shape),), input_shape, ns)
        self.seed = seed

        # general params
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W: np.ndarray = np.array([])
        self.b: np.ndarray = np.array([])
        self.__input_shape: tuple = input_shape

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.reg: Norm = reg
        self.opt: Optimizer() = opt if opt else Adam()
        self.opt_bias_: Optimizer() = opt_bias_ if opt_bias_ or not bias else Adam()

        self.init_weights()

    def forward(self, X: np.ndarray) -> np.ndarray:
        W, b, X = self.W, self.b, self.reshape(X)

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
        dReg, delta, X = self.reg.d_norm, self.reshape(delta, in_=False), self.reshape(X)
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
        if not self.reg or self.lambda_ == 0:
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
    def input_shape(self, input_shape_: Union[int, tuple]):
        self.__input_shape = (int(np.prod(input_shape_)),)
        self.init_weights()

    def init_weights(self):
        if self.__input_shape and self.__input_shape[0]:
            bias, eps = self.bias, self.eps
            # TODO remove
            if self.seed >= 0:
                np.random.seed(self.seed)
            self.W = self.init_W((self.__input_shape[0], self.out_shape[0]), eps)
            if bias:
                self.b = np.zeros((self.out_shape[0],))
                # self.b = self.init_W((self.out_shape[0],), eps)

    def reshape(self, X, in_=True):
        return np.reshape(X, (-1,) + (self.input_shape if in_ else self.out_shape))


class NormLayer(Layer):
    """
    if axis=0: normalize per attribute
    if axis=1: normalize per training example
    """

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None, eps=1e-5,
                 momentum=.9, beta_opt: Optimizer = None, gamma_opt: Optimizer = None, alpha: float = 1e-7,
                 mode: str = 'batch', transpose_order: Union[tuple, np.ndarray] = None, data_type: str = '',
                 group_size: int = None, group_axis: int = None) -> None:
        super().__init__(out_shape, input_shape, False)
        # general params
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.__input_shape = self.out_shape = input_shape
        self.mode: str = mode
        self.axis: Union[int, tuple] = 0
        self.trans_order: np.ndarray = transpose_order
        self.revers_trans: np.ndarray = transpose_order
        self.group_size: int = group_size
        self.group_axis: int = group_axis

        # special case for RGB
        if data_type.upper() == 'RGB' and transpose_order is None:
            if mode.lower() == 'batch':
                self.trans_order = np.array([0, 2, 3, 1])
        if data_type.upper() == 'RGB' and mode.lower() == 'group' and not self.group_axis:
            self.axis = (2, 3, 4)
            self.group_axis = 1

        # reg params
        self.mu = 0
        self.var = 0
        self.std = 0
        self.z = 0
        self.eps = eps
        self.alpha = alpha
        self.momentum = momentum

        self.gamma: np.ndarray = np.array([])
        self.beta: np.ndarray = np.array([])
        self.running_mu: np.ndarray = np.array([])
        self.running_var: np.ndarray = np.array([])

        # opt params
        self.beta_opt: Optimizer() = beta_opt if beta_opt else Adam()
        self.gamma_opt: Optimizer() = gamma_opt if gamma_opt else Adam()

        self.init_weights()

    def forward(self, X: np.ndarray, mode='train') -> np.ndarray:
        gamma, beta, layer_norm, eps, axis, norm_type = self.gamma, self.beta, self.axis, self.eps, self.axis, self.mode
        keep_dims = norm_type.lower() == 'group'

        X = self.reshape_transpose(X, 'before')

        if norm_type != 'batch' or mode == 'train':
            mu, var = X.mean(axis=axis, keepdims=keep_dims), X.var(axis=axis, keepdims=keep_dims) + eps
            std = var ** 0.5

            if axis == 1:
                X = X.T.copy()

            z = (X - mu) / std
            # z = (X - mu) / (std + eps)#check version
            if norm_type == 'group':
                z = z.reshape((-1, *self.input_shape))

            out = gamma * z + beta

            if norm_type.lower() == 'batch':
                momentum = self.momentum
                self.running_mu = momentum * self.running_mu + (1 - momentum) * mu
                self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.std, self.var, self.mu, self.z = std, var, mu, z

        elif mode == 'test':
            out = gamma * (X - self.running_mu) / (self.running_var + eps) ** 0.5 + beta

        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)

        if norm_type.lower() == 'layer':
            out = out.T

        out = self.reshape_transpose(out, 'after')

        return out

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> Union[np.ndarray, tuple]:
        grades, alpha = self.grad(delta, X), self.alpha
        d_beta, d_gamma, delta = grades['d_beta'], grades['d_gamma'], grades['delta']

        self.beta -= self.beta_opt.opt(d_beta, alpha)
        self.gamma -= self.gamma_opt.opt(d_gamma, alpha)

        return delta

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        z, gamma, beta, mu, var, std, axis = self.z, self.gamma, self.beta, self.mu, self.var, self.std, self.axis
        norm_type = self.mode.lower()
        keep_dims = norm_type == 'group'

        if norm_type.lower() == 'batch' or norm_type.lower() == 'layer':
            delta = self.reshape_transpose(delta, 'before')
        elif norm_type.lower() == 'group':
            axis = (0,) + axis[:-1]

        d_beta = delta.sum(axis=axis, keepdims=keep_dims)
        d_gamma = np.sum(delta * z, axis=axis, keepdims=keep_dims)

        df_dz = delta * gamma

        if norm_type.lower() == 'group':
            z = self.reshape_transpose(z, 'before')
            df_dz = self.reshape_transpose(df_dz, 'before')
            m = int(np.prod(z.shape[self.group_axis + 1:]))
            axis = self.axis
            delta = (1 / (m * std)) * (
                    m * df_dz - np.sum(df_dz, axis=axis, keepdims=keep_dims) - z * np.sum(df_dz * z, axis=axis,
                                                                                          keepdims=keep_dims))
        else:
            m = delta.shape[0]
            delta = (1 / (m * std)) * (m * df_dz - np.sum(df_dz, axis=0) - z * np.sum(df_dz * z, axis=0))

        if norm_type.lower() == 'layer':
            delta, d_beta, d_gamma = delta.T, d_beta.reshape(beta.shape), d_gamma.reshape(gamma.shape)

        delta = self.reshape_transpose(delta, 'after')

        return {'delta': delta, 'd_beta': d_beta, 'd_gamma': d_gamma}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def reshape_transpose(self, X, mode='before'):
        norm_type = self.mode

        if mode == 'before':
            if self.revers_trans is not None:
                X = X.transpose(self.trans_order).copy()
            if (norm_type.lower() == 'batch' or norm_type.lower() == 'layer') and len(self.input_shape) > 1:
                X = X.reshape((-1, self.input_shape[0])).copy()
            elif norm_type.lower() == 'group':
                g = self.group_size
                shape = (-1,) + (g, +self.__input_shape[0] // g, *self.__input_shape[1:])
                X = X.reshape(shape)

        elif mode == 'after':
            if len(self.input_shape) > 1:
                shape = (-1,) + self.input_shape
                if self.revers_trans is not None:
                    shape = tuple(shape[i] for i in self.trans_order)
                X = X.reshape(shape)

            if self.revers_trans is not None:
                X = X.transpose(self.revers_trans)

        return X

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        if isinstance(input_shape_, int):
            input_shape_ = (input_shape_,)
        self.__input_shape = self.out_shape = input_shape_
        self.init_weights()

    def init_weights(self):
        if self.__input_shape is None:
            return

        shape, mode = self.__input_shape, self.mode

        size = self.out_shape[0]
        self.gamma: np.ndarray = np.ones(size, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(size, dtype=np.float64)

        if mode.lower() == 'batch':
            self.running_mu: np.ndarray = np.zeros(size, dtype=np.float64)
            self.running_var: np.ndarray = np.zeros(size, dtype=np.float64)
        if mode.lower() == 'layer':
            self.gamma = self.gamma.reshape((-1, 1))
            self.beta = self.beta.reshape((-1, 1))
        elif mode.lower() == 'group':
            new_shape = [1 for _ in shape]
            new_shape.insert(self.group_axis, size)
            new_shape = tuple(new_shape)
            self.gamma = self.gamma.reshape(new_shape)
            self.beta = self.beta.reshape(new_shape)

        if self.trans_order is not None:
            self.trans_order = np.array([*self.trans_order])
            a = np.arange(len(self.input_shape) + 1)
            move = self.trans_order - a
            self.revers_trans = self.trans_order[a + move[a]]

    def reshape(self, data, in_=True):
        pass


class Dropout(Layer):

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None, ns: bool = False,
                 p: float = 0.5, seed=-1) -> None:
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

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        self.__input_shape = self.out_shape = input_shape_

    def reshape(self, data, in_=True):
        pass


class SoftmaxStableLayer(ActLayer):

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None,
                 ns: bool = True) -> None:
        super().__init__(out_shape, input_shape, ns, SoftmaxStable())
        self.log_prob = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        if self.ns:
            Z -= np.max(Z, axis=1, keepdims=True)
        self.log_prob = self.act.activation(Z)
        return np.exp(self.log_prob)

    def grad(self, delta: np.ndarray, X: np.ndarray, H: np.ndarray = None, return_delta=True) -> dict[str:np.ndarray]:
        return super().grad(delta, X)

    def backward(self, delta: np.ndarray, X: np.ndarray, H: np.ndarray = None, return_delta=True) -> np.ndarray:
        return super().backward(delta, X, H, return_delta)

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return super().delta(y, pred)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return super().loss(y, self.log_prob)


class Conv(Layer):

    def __init__(self, out_shape: tuple = None, input_shape: Union[int, tuple] = None, ns: bool = False,
                 filter_shape: tuple = (5, 5), filter_num: int = 3, stride: int = 1, pad_h: int = None,
                 pad_w: int = None,
                 pad_type: float = 0, act=np.sum, opt: Optimizer = None, opt_bias: Optimizer = None,
                 alpha: float = 1e-7, lambda_: float = .0, reg: Norm = Norm2, eps: float = 1e-3,
                 init_W: InitWeights = xavierScale, bias: bool = True, reg_bias: bool = False) -> None:
        super().__init__(out_shape, input_shape, ns)

        # general params
        self.stride: int = stride
        self.pad_h: int = pad_h if pad_h is not None else (filter_shape[0] - 1) // 2
        self.pad_w: int = pad_w if pad_w is not None else (filter_shape[1] - 1) // 2
        self.pad_type: float = pad_type
        self.filter_shape: tuple = filter_shape
        self.filter_num: int = filter_num
        self.__input_shape: tuple = input_shape
        self.out_shape = None
        if input_shape:
            h, w = input_shape[1:]
            f = filter_num
            fh, fw = filter_shape[0], filter_shape[1]
            h_o, w_o = 1 + (h - fh + 2 * self.pad_h) // stride, 1 + (w - fw + 2 * self.pad_w) // stride
            self.out_shape = (f, h_o, w_o)

        # weighs
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.W: np.ndarray = np.array([])
        self.b: np.ndarray = np.array([])
        self.Xp: np.ndarray = np.array([])

        self.act = act
        # optimizer params
        self.eps: float = eps
        self.init_W: InitWeights = init_W
        self.alpha: float = alpha
        self.lambda_: float = lambda_
        self.reg: Norm = reg
        self.opt: Optimizer() = opt if opt else Adam()
        self.opt_bias: Optimizer() = opt if opt else Adam()
        self.bias_opt: Optimizer() = opt_bias if opt_bias else Adam()

        self.init_weights()

    def forward(self, X: np.ndarray) -> np.ndarray:
        W, b, pad_h, pad_w, stride = self.W, self.b, self.pad_h, self.pad_w, self.stride
        X = self.reshape(X)

        m, _, h, w = X.shape
        f, _, fh, fw = self.W.shape
        assert (h - fh + pad_h * 2) % stride == 0 and (w - fw + pad_w * 2) % stride == 0
        h_o, w_o = 1 + (h - fh + 2 * pad_h) // stride, 1 + (w - fw + 2 * pad_w) // stride
        Xp = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
        out = np.empty((m, f, h_o, w_o))

        # slow
        # for example in range(m):
        #     for filter_ in range(f):
        #         for row in range(h_o):
        #             for col in range(w_o):
        #                 x = Xp[example, :, row * stride:row * stride + fh, col * stride:col * stride + fw]
        #                 out[example, filter_, row, col] = np.sum(x * W[filter_])  # +b[filter_]

        # fast x80
        hh = 0
        for i in range(h_o):
            ww = 0
            for j in range(w_o):
                out[:, :, i, j] = np.tensordot(Xp[:, :, hh:hh + fh, ww:ww + fw], W, axes=[(1, 2, 3), (1, 2, 3)])
                ww += stride
            hh += stride

        out += b[None, :, None, None]

        self.Xp = Xp
        return out

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        W, b, alpha = self.W, self.b, self.alpha
        grades = self.grad(delta, X)
        dW, db = grades['dW'], grades['db']

        W -= self.opt.opt(dW, alpha)
        b -= self.opt_bias.opt(b, alpha)

        return grades['delta']

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        W, b, Xp, pad_h, pad_w, stride = self.W, self.b, self.Xp, self.pad_h, self.pad_w, self.stride
        X, delta = self.reshape(X), self.reshape(delta, in_=False)

        m, _, h, w = X.shape
        f, _, fh, fw = self.W.shape
        assert (h - fh + pad_h * 2) % stride == 0 and (w - fw + pad_w * 2) % stride == 0
        h_o, w_o = 1 + (h - fh + 2 * pad_h) // stride, 1 + (w - fw + 2 * pad_w) // stride

        dW = np.zeros_like(W)
        dXp = np.zeros_like(Xp)
        db = np.sum(delta, axis=(0, 2, 3))

        # slow
        # for m_i in range(m):
        #     for f_i in range(f):
        #         for r in range(h_o):
        #             for c in range(w_o):
        #                 s = stride
        # dXp[m_i, :, r * s:r * s + fh, c * s:c * s + fw] += W[f_i] * delta[m_i, f_i, r, c]
        # dW[f_i] += Xp[m_i, :, r * s:r * s + fh, c * s:c * s + fw] * delta[m_i, f_i, r, c]

        # fast x100

        hh = 0
        for i in range(h_o):
            ww = 0
            for j in range(w_o):
                delta_i = delta[:, :, i, j]
                dW += np.tensordot(delta_i, Xp[:, :, hh:hh + fh, ww:ww + fw], axes=((0,), (0,)))
                dXp[:, :, hh:hh + fh, ww:ww + fw] += np.tensordot(delta_i, W, axes=((1,), (0,)))
                ww += stride
            hh += stride

        delta = dXp[:, :, pad_h:-pad_h, pad_w:-pad_w]

        dReg, lambda_ = self.reg.d_norm, self.lambda_
        return {'delta': delta, 'dW': dW + lambda_ * dReg(W) / 2, 'db': db + lambda_ * dReg(b) / 2}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

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
    def input_shape(self, input_shape):
        self.__input_shape = input_shape
        self.init_weights()

        h, w = self.input_shape[1:]
        f = self.filter_num
        stride, fh, fw, pad_h, pad_w = self.stride, self.filter_shape[0], self.filter_shape[1], self.pad_h, self.pad_w
        h_o, w_o = 1 + (h - fh + 2 * pad_h) // stride, 1 + (w - fw + 2 * pad_w) // stride
        self.out_shape = (f, h_o, w_o)

    def init_weights(self):
        self.b = np.zeros(self.filter_num)
        if self.__input_shape:
            self.W = self.init_W((self.filter_num, self.__input_shape[0], *self.filter_shape), self.eps)

    def reshape(self, X, in_=True):
        return np.reshape(X, (-1,) + (self.input_shape if in_ else self.out_shape))


class MaxPooling(Layer):

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: tuple = None, ns: bool = False,
                 kernel_shape: tuple = (2, 2), stride: int = 1) -> None:
        super().__init__(out_shape, input_shape, ns)
        self.kernel_shape: tuple = kernel_shape
        self.stride: int = stride
        self.__input_shape: Union[int, tuple] = input_shape
        self.init()

    def forward(self, X: np.ndarray) -> np.ndarray:
        m, c, h, w = X.shape
        stride, hp, wp = self.stride, self.kernel_shape[0], self.kernel_shape[1]
        h_o, w_o = 1 + (h - hp) // stride, 1 + (w - wp) // stride
        out = np.empty((m, c, h_o, w_o))

        # fast x5
        hh = 0
        for i in range(h_o):
            ww = 0
            for j in range(w_o):
                out[:, :, i, j] = np.max(X[:, :, i * stride:i * stride + hp, j * stride:j * stride + wp], axis=(2, 3))
                ww += stride
            hh += stride

        # slow
        # for m_i in range(m):
        #     for c_i in range(c):
        #         for r in range(h_o):
        #             for col in range(w_o):
        #                 out[m_i, c_i, r, col] = np.max(
        #                     X[m_i, c_i, r * stride:r * stride + hp, col * stride:col * stride + wp])

        return out

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        return self.grad(delta, X)['delta']

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        # m, c, h, w = X.shape
        stride, hp, wp = self.stride, self.kernel_shape[0], self.kernel_shape[1]
        delta = self.reshape(delta)
        # h_o, w_o = 1 + (h - hp) // stride, 1 + (w - wp) // stride
        # delta_ = np.empty(X.shape)
        # slow
        # for m_i in range(m):
        #     for c_i in range(c):
        #         for r in range(h_o):
        #             for col in range(w_o):
        #                 x = X[m_i, c_i, r * stride:r * stride + hp, col * stride:col * stride + wp]
        #                 idx_ = np.unravel_index(np.argmax(x), x.shape)
        #                 delta_[m_i, c_i, r * stride + idx_[0], col * stride + idx_[1]] = delta[m_i, c_i, r, col]

        # fast x
        m, c, h, w = X.shape
        h_o = 1 + (h - hp) // stride
        w_o = 1 + (w - wp) // stride

        delta_ = np.empty(X.shape)
        hh, ww = 0, 0

        for i in range(h_o):
            ww = 0
            for j in range(w_o):
                x = X[:, :, hh:hh + hp, ww:ww + wp]
                # x_max = np.max(x, axis=(2, 3))
                mask = (x == np.max(x, axis=(2, 3))[:, :, None, None])
                delta_[:, :, hh:hh + hp, ww:ww + wp] = mask * delta[:, :, i, j][:, :, None, None]
                ww += stride
            hh += stride

        return {'delta': delta_}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        if input_shape_:
            self.__input_shape = input_shape_
            self.init()

    def init(self):
        if self.input_shape:
            hp, wp = self.kernel_shape
            c, h, w = self.__input_shape
            stride = self.stride
            h_o, w_o = 1 + (h - hp) // stride, 1 + (w - wp) // stride
            self.out_shape = (c, h_o, w_o)

    def reshape(self, X, in_=False):
        return np.reshape(X, (-1,) + (self.input_shape if in_ else self.out_shape))


class VanillaRNN(Layer):

    def __init__(self, input_shape: Union[int, tuple] = None,
                 reg: Norm = Norm2,
                 ns: bool = False, opt_x: Optimizer = None, opt_h: Optimizer = None, opt_bias_: Optimizer = None,
                 eps=1e-3, alpha=1e-5,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W: InitWeights = xavierScale,
                 act=None) -> None:
        self.__input_shape = input_shape
        super().__init__(None, input_shape, ns)

        # general params
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.Wx: np.ndarray = np.array([])
        self.Wh: np.ndarray = np.array([])
        self.b: np.ndarray = np.array([])
        self.H: np.ndarray = np.array([])
        self.t: int = 1
        self.__input_shape: tuple = input_shape

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = act if act else Tanh()
        self.reg: Norm = reg
        self.opt_x: Optimizer() = opt_x if opt_x else Adam()
        self.opt_h: Optimizer() = opt_h if opt_h else Adam()
        self.opt_bias_: Optimizer() = opt_bias_ if opt_bias_ or not bias else Adam()

        self.init()

    def forward(self, data: tuple) -> np.ndarray:
        X, h = self.reshape(data)

        Wx, Wh, b, Act = self.Wx, self.Wh, self.b, self.act.activation
        m, t, w = X.shape

        H = self.H = np.empty((m, t, b.shape[0]))

        for i in range(t):
            x = X[:, i, :]
            Wx_x, Wh_h = x @ Wx, h @ Wh
            z = Wx_x + Wh_h + b
            self.H[:, i, :] = z
            h = H[:, i, :] = Act(z)

        return H

    def backward(self, delta: np.ndarray, data: tuple, return_delta=True) -> list:
        Wx, Wh, b, grades, alpha = self.Wx, self.Wh, self.b, self.grad(delta, data), self.alpha
        dWx, dWh = grades['dWx'], grades['dWh']

        Wx -= self.opt_x.opt(dWx, alpha)
        Wh -= self.opt_h.opt(dWh, alpha)
        if self.bias:
            db = grades['db']
            b -= self.opt_bias_.opt(db, self.alpha)

        return grades['delta']

    def grad(self, delta: np.ndarray, data: tuple) -> dict[str:np.ndarray]:
        delta = self.reshape(delta, False)
        m, t, _ = delta.shape
        X, h0 = self.reshape(data)
        Wx, Wh, b, n, H, dAct = self.Wx, self.Wh, self.b, X.shape[1], self.H, self.act.grad
        delta_x, dWx, dWh, db = np.zeros(X.shape), np.zeros(Wx.shape), np.zeros(Wh.shape), np.zeros(b.shape)

        dh_i = 0
        for i in reversed(range(t)):
            if i == 0:
                prev_h = h0
            else:
                prev_h = H[:, i - 1, :]
            da = dAct(H[:, i, :]) * (delta[:, i, :] + dh_i)
            # da = (1 - H[:, i, :] ** 2) * (delta[:, i, :] + dh_i)

            dWx_i = X[:, i, :].T @ da
            dWh_i = prev_h.T @ da
            db_i = np.sum(da, axis=0)
            dx = da @ Wx.T
            d_prev_h = da @ Wh.T

            delta_x[:, i, :], dh_i = dx, d_prev_h
            dWh, dWx, db = dWh + dWh_i, dWx + dWx_i, db + db_i
        delta_h0 = dh_i
        return {'delta': [delta_x, delta_h0], 'dWx': dWx, 'dWh': dWh, 'db': db}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple):
        self.__input_shape = input_shape_
        self.init()

    def init(self) -> None:
        if self.input_shape:
            assert len(self.input_shape) == 2
            self.out_shape = (np.prod(self.input_shape[1]),)
            bias, eps = self.bias, self.eps

            self.Wx = self.init_W((np.prod(self.__input_shape[0]), self.out_shape[0]), eps)
            self.Wh = self.init_W((self.out_shape[0], self.out_shape[0]), eps)
            if bias:
                self.b = np.zeros((self.out_shape[0],))
                # self.b = self.init_W((self.out_shape[0],), eps)

    def reshape(self, data, in_=True) -> Union[tuple, np.ndarray]:
        if in_:
            if len(data) == 2:
                X, h = data[0], data[1]
                self.t = X.shape[1]
                return X, h
            else:  # test mode
                return data[0], self.H[0]
        else:
            data = np.reshape(data, (-1,) + (self.t, *self.out_shape))
            return data


class LstmRNN(VanillaRNN):

    def __init__(self, input_shape: Union[int, tuple] = None, reg: Norm = Norm2, ns: bool = False,
                 opt_x: Optimizer = None, opt_h: Optimizer = None, opt_bias_: Optimizer = None, eps=1e-3, alpha=1e-5,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W: InitWeights = xavierScale,
                 act: list = None) -> None:
        self.__input_shape = input_shape
        self.act: list = [Sigmoid(), Sigmoid(), Sigmoid(), Tanh()] if not act else act
        super().__init__(input_shape, reg, ns, opt_x, opt_h, opt_bias_, eps, alpha, lambda_, bias, reg_bias, opt_bias,
                         init_W, self.act)
        self.C_tanh = None
        self.C = None
        self.H = None
        self.gates = None
        self.init()

    def forward(self, data: tuple) -> np.ndarray:
        X, h_prev = data

        m, t, d = X.shape
        _, h = h_prev.shape
        Wx, Wh, b = self.Wx, self.Wh, self.b

        H, C = np.empty((m, t, h)), np.empty((m, t, h))
        H[:, 0, :], C[:, 0, :] = h_prev, 0
        gates, C_tanh = np.empty((t, len(self.act), m, h)), np.empty((t, m, h))

        if t > 0:
            H[:, 0, :], C[:, 0, :], gates[0], C_tanh[0] = self.step_forward(X[:, 0, :], h_prev, 0, Wx, Wh, b)

        for i in range(1, t):
            H[:, i, :], C[:, i, :], gates[i], C_tanh[i] = self.step_forward(X[:, i, :], H[:, i - 1, :], C[:, i - 1, :],
                                                                            Wx, Wh, b)

        self.C, self.C_tanh, self.H, self.gates, self.t = C, C_tanh, H, gates, t
        return H

    def step_forward(self, step_x, prev_h, prev_c, Wx, Wh, b) -> tuple:
        x, h, c = step_x, prev_h, prev_c

        Wx_x, Wh_h = x @ Wx, h @ Wh
        z = Wx_x + Wh_h + b

        size = Wh.shape[0]
        gates = i, f, o, g = [self.act[j].activation(z[:, size * j:size * (j + 1)]) for j in range(len(self.act))]

        c = f * c + i * g
        c_tanh = self.act[-1].activation(c)
        h = o * c_tanh
        return h, c, np.array(gates), c_tanh

    @staticmethod
    def step_backward(delta: tuple, data: tuple, gates, c_tanh, Wx, Wh):
        d_h, d_c = delta
        x, prev_h, prev_c = data
        N, H = d_h.shape
        da = np.empty((N, 4 * H))
        tanh_next_c = c_tanh
        i, f, o, g = gates

        do = tanh_next_c * d_h
        d_c += o * (1 - tanh_next_c ** 2) * d_h

        df = d_c * prev_c
        d_prev_c = f * d_c
        di = d_c * g
        dg = i * d_c

        dag = (1 - g ** 2) * dg
        da[:, 3 * H:4 * H] = dag

        dao = o * (1 - o) * do
        da[:, 2 * H:3 * H] = dao

        daf = f * (1 - f) * df
        da[:, 1 * H:2 * H] = daf

        dai = i * (1 - i) * di
        da[:, 0 * H:1 * H] = dai

        dWx_x = da
        dWh_h = da
        db = np.sum(da, axis=0)

        dWx = x.T @ dWx_x
        dx = dWx_x @ Wx.T

        dWh = prev_h.T @ dWh_h
        d_prev_h = dWh_h @ Wh.T
        return dx, d_prev_h, d_prev_c, dWx, dWh, db

    def grad(self, delta: np.ndarray, data: tuple) -> dict[str:np.ndarray]:
        X, h0 = data
        m, t, d = X.shape
        _, h = h0.shape

        delta = self.reshape(delta, False)
        Wx, Wh, b, H, gates, C, C_tanh = self.Wx, self.Wh, self.b, self.H, self.gates, self.C, self.C_tanh
        delta_x = np.empty((m, t, d))
        dWx = np.zeros((d, 4 * h))
        dWh = np.zeros((h, 4 * h))
        db = np.zeros((4 * h,))

        d_h, d_c = 0, 0
        for i in reversed(range(t)):
            if i == 0:
                h_prev, c_prev = h0, 0
            else:
                h_prev, c_prev = H[:, i - 1, :], C[:, i - 1, :]

            delta_x[:, i, :], d_h, d_c, dWx_i, dWh_i, db_i = \
                self.step_backward((delta[:, i, :] + d_h, d_c), (X[:, i, :], h_prev, c_prev), gates[i], C_tanh[i],
                                   Wx, Wh)
            dWx += dWx_i
            dWh += dWh_i
            db += db_i

        delta_h0 = d_h
        return {'delta': (delta_x, delta_h0), 'dWx': dWx, 'dWh': dWh, 'db': db}

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple):
        self.__input_shape = input_shape_
        self.init()

    def init(self) -> None:
        if self.input_shape:
            assert len(self.input_shape) == 2
            self.out_shape = (np.prod(self.input_shape[1]),)
            bias, eps = self.bias, self.eps

            self.Wx = self.init_W((np.prod(self.__input_shape[0]), len(self.act) * self.out_shape[0]), eps)
            self.Wh = self.init_W((self.out_shape[0], len(self.act) * self.out_shape[0]), eps)
            if bias:
                self.b = np.zeros((len(self.act) * self.out_shape[0],))
                # self.b = self.init_W((self.out_shape[0],), eps)

    def reshape(self, data, in_=True) -> Union[tuple, np.ndarray]:
        if in_:
            if len(data) == 3:
                X, h, c = data[0], data[1], data[2]
                return X, h, c
            else:  # test mode
                return data[0], self.H[0]
        else:
            data = np.reshape(data, (-1,) + (self.t, *self.out_shape))
            return data


class EmbedLayer(WeightLayer):

    def __init__(self, out_shape: Union[int, tuple], input_shape: Union[int, tuple], reg: Norm = None,
                 ns: bool = False, opt: Optimizer = None, opt_bias_: Optimizer = None, eps=0.01, alpha=1e-7, lambda_=0,
                 bias=False, reg_bias=False, opt_bias=False, init_W: InitWeights = stdScale, seed=-1) -> None:
        super().__init__(out_shape, input_shape, reg, ns, opt, opt_bias_, eps, alpha, lambda_, bias, reg_bias, opt_bias,
                         init_W, seed)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: array of indexes for every train example (m X t)
            where t is the max of sequence length

        :return: (m X t X d) weights for every example in every sequence we have weights for
            all the dimensions
        """
        return self.W[X, :]

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=False) -> None:
        dW = self.grad(delta, X)['dW']
        self.W -= self.opt.opt(dW, self.alpha)

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        dW = np.zeros(self.W.shape)
        np.add.at(dW, X, delta)
        return {'dW': dW}


class ReshapeLayer(Layer):
    """
    layer for reshape
    """

    def __init__(self, out_shape: Union[int, tuple] = None, input_shape: Union[int, tuple] = None) -> None:
        super().__init__(out_shape, input_shape)
        self.__input_shape = input_shape

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.reshape(X, (-1,) + self.out_shape)

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> np.ndarray:
        return self.grad(delta, X)['delta']

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict[str:np.ndarray]:
        return {'delta': np.reshape(delta, (-1,) + self.input_shape)}

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_):
        self.__input_shape = input_shape_

    def reshape(self, data, in_=True):
        pass


class ConnectLayer(Layer):

    def __init__(self, layers: list[Layer], out_shape: Union[int, tuple] = None,
                 input_shape: Union[int, tuple] = None) -> None:
        super().__init__(out_shape, input_shape)
        self.layers: list[Layer] = layers
        self.__input_shape = input_shape
        self.init()

    def forward(self, data: tuple[np.ndarray]) -> list:
        out = []
        for layer, X in zip(self.layers, data):
            out.append(layer.forward(X))
        return out

    def backward(self, delta: tuple, data: tuple, return_delta=True) -> Union[None, list[np.ndarray]]:
        next_delta = []
        for layer, delta_, X in zip(self.layers, delta, data):
            d_i = layer.backward(delta_, X)
            if d_i is not None:
                next_delta.append(d_i)
        if return_delta:
            return next_delta if len(next_delta) > 1 else next_delta[0]

    def grad(self, delta: tuple, data: tuple) -> list[dict[str:np.ndarray]]:
        grades = []
        for layer, delta_, X in zip(self.layers, delta, data):
            grades.append(layer.grad(delta_, X))
        return grades

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def reshape(self, data, in_=True):
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_):
        self.init()

    def init(self) -> None:
        input_shape, out_shape = [], []
        for layer in self.layers:
            input_shape.append(layer.input_shape), out_shape.append(layer.out_shape)
        self.__input_shape, self.out_shape = tuple(input_shape), tuple(out_shape)


class ConnectEmbedWeightRNN(ConnectLayer):

    def __init__(self, embed_layer: EmbedLayer, weight_layer: WeightLayer) -> None:
        input_shape = (embed_layer.input_shape, weight_layer.input_shape)
        out_shape = (embed_layer.out_shape, weight_layer.out_shape)
        self.embed_layer: EmbedLayer = embed_layer
        self.weight_layer: WeightLayer = weight_layer

        super().__init__([embed_layer, weight_layer], out_shape, input_shape)

        self.__input_shape = (self.embed_layer.input_shape, self.weight_layer.input_shape)
        self.init()

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_):
        self.weight_layer.input_shape = input_shape_
        self.init()
