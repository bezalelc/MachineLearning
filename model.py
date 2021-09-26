import numpy as np
from regularization import Regularization, L2
from activation import Activation, Softmax
from optimizer import Optimizer, Vanilla
from layer import Layer, Dense, WeightLayer, NormLayer, ActLayer, ComplexLayer, Dropout
import metrics
import time


class Model:
    def __init__(self) -> None:
        super().__init__()
        # general param
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        # hyper param
        self.threshold = 0.5
        # self.reg: Regularization() = None  # Regularization function
        # self.dReg: dRegularization = None  # derivative for regularization function
        # model param
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def compile(self) -> None:
        """
        restart hyper params
        """
        pass
        # self.reg = reg
        # self.reg, self.d_reg, self.activation, self.loss_ = reg, d_reg, activation, loss_

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.m = X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def loss(self, X: np.ndarray, y: np.ndarray):
        pass

    def split(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def describe(self):
        pass

    def k_fold(self, X, y, k: int = 5):
        """
        split dataset to k folders and train k times while in each loop the fold[i] if the validation
            data, this method is useful for a small dataset

        :param X:
        :param y:
        :param k:
        :return:
        """
        X, y = np.array_split(X, k), np.array_split(y, k)

        for i in range(k):
            X_, y_ = np.vstack([X[j] for j in range(k) if j != i]), np.hstack([y[j] for j in range(k) if j != i])
            self.train(X_, y_)


class KNearestNeighbor(Model):

    def __init__(self, reg: Regularization) -> None:
        super().__init__()
        self.reg = reg

    def train(self, X: np.ndarray, y: np.ndarray):
        super().train(X, y)
        self.m, self.n = X.shape[:2]

    def predict(self, X_test: np.ndarray, k: int = 1) -> np.ndarray:
        """
        k nearest neighbors algorithm:
            predict x according to the closest distance values
            this function is for classification predict


        :param X_test: data to predict
        :param k: range
        :param reg_func:  function for regularization

        :return: prediction

        :efficiency: O(m*n*test_size)
        """

        n, n_test, Reg = self.X.shape[0], X_test.shape[0], self.reg.norm
        distances = np.empty((n_test, n))
        for i in range(n_test):
            for j in range(n):
                distances[i, j] = Reg(X_test[i] - self.X[j]) ** 0.5
        # for i in range(n_test):
        # distances[i, :] = np.sum(self.reg.norm(X_test[i] - self.X)) ** 0.5
        # distances[i, :] = np.sum((X_test[i] - self.X) ** 2, axis=1) ** 0.5

        # distances = np.sum(Reg(X_test[:, np.newaxis] - self.X), axis=2) ** 0.5
        idx = np.argpartition(distances, k, axis=1)[:, :k].reshape((-1, k))
        neighbor = self.y[idx].reshape((-1, k))

        from scipy.stats import mode
        pred = mode(neighbor, axis=1)[0]
        return pred


class Regression(Model):

    def __init__(self, layers: list[Layer] = None, classes: list = None) -> None:
        super().__init__()

        # model param
        self.layers: list[Layer] = layers
        # self.layers_size: list = []

        # general params
        self.classes: list = classes

    def compile(self) -> None:
        layers = self.layers
        assert layers is not None and len(layers) > 0

        self.n, self.k = layers[0].input_shape[0], layers[-1].out_shape
        for i, layer in enumerate(layers):
            # self.layers_size.append((layer.input_shape, layer.out_shape))

            if i > 0:  # and (isinstance(layer, Dense) or isinstance(layer, Dense))
                layer.input_shape = (layers[i - 1].out_shape,)

    def train(self, X: np.ndarray, y: np.ndarray, val: tuple = None, iter_=1500, batch=32, epoch=100, return_loss=True,
              verbose=True, lrd=0.95, acc_func=metrics.accuracy) -> tuple[list, list]:
        super().train(X, y)  # np.hstack((np.ones((X.shape[0], 1)), X))
        # unpacked param
        m, n, k, layers = X.shape[0], X.shape[1], self.k, self.layers
        batch, epoch = min(batch, m), min(epoch, iter_)
        loss_history_t, loss_history_v = [], []
        H_val = None

        for i in range(iter_):
            batch_idx = np.random.choice(m, batch, replace=False)
            X_, y_ = X[batch_idx], y[batch_idx]

            H = self.feedforward(X_)
            self.backpropagation(H, y_)
            time.sleep(2)

            if return_loss:
                loss_history_t.append(self.loss(X_, y_, mode='test'))
                if val:
                    H_val = self.feedforward(val[0], mode='test')
                    loss_history_v.append(self.loss(*val, H_val[-1], mode='test'))
            if verbose and i % epoch == 0:
                s = 'iteration %d / %d: loss %f acc %f' % (
                    i + 1, iter_, loss_history_t[-1],
                    acc_func(y_, self.predict(X_, self.feedforward(X_, mode='test')[-1])))
                if val:
                    s += ' val loss %f val acc %f' % \
                         (loss_history_v[-1], acc_func(val[1], self.predict(val[0], H_val[-1])))
                print(s)

        return loss_history_t, loss_history_v

    def feedforward(self, X: np.ndarray, mode='train') -> list[np.ndarray]:
        layers = self.layers

        H = [X]
        for layer in layers:
            if isinstance(layer, (NormLayer, Dropout)):
                H.append(layer.forward(H[-1], mode=mode))
            else:
                H.append(layer.forward(H[-1]))

        return H

    def backpropagation(self, H, y, mode='train'):
        layers = self.layers

        delta = self.layers[-1].delta(y, H[-1])
        for layer, h, i in zip(layers[:-1][::-1], H[:-2][::-1], range(len(layers[:-1]))[::-1]):
            if isinstance(layer, (NormLayer, Dropout)):
                delta = layer.backward(delta, h, return_delta=bool(i), mode=mode)
            else:
                delta = layer.backward(delta, h, return_delta=bool(i))

    def grad(self, H: list[np.ndarray], y: np.ndarray) -> list[dict]:
        layers, grades = self.layers, []

        grades.append({'delta': self.layers[-1].delta(y, H[-1])})
        for layer, h, i in zip(layers[:-1][::-1], H[:-2][::-1], range(len(layers[:-1][::-1]))):
            grades.append(layer.grad(grades[-1]['delta'], h))

        return grades[::-1]

    def loss(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray = None, mode='train') -> float:
        if pred is None:
            pred = self.feedforward(X, mode=mode)[-1]  # , mode='train'
        m, Loss, layers = pred.shape[0], self.layers[-1].loss, self.layers
        J = Loss(y, pred)

        for layer in layers:
            if isinstance(layer, (Dense, WeightLayer)):
                J += layer.regularize() / 2
        # J += sum(layer.regularize() if isinstance(layer, Dense) else 0 for layer in layers) / 2
        # J += sum(layer.regularize() for layer in layers)
        return J

    def predict(self, X: np.ndarray, pred: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
        if pred is None:
            pred = self.feedforward(X, mode='test')[-1]
        return np.argmax(pred, axis=1)

    def __iadd__(self, other: Layer):
        if isinstance(other, Dense):
            other.input_shape = (self.layers[-1].out_shape,)

        self.k = other.out_shape
        self.layers.append(other)
        return self

    def describe(self):
        layers = self.layers
        for i, layer in enumerate(layers):
            print(f'{i}. {layer.W.shape}')

    def best_alpha_lambda(self, X, y, Xv, Yv, alphas, lambdas, verbose=True):
        """
        choose the best hyper params alpha and lambda for svm

        :param X: train data
        :param y: classes for train data
        :param Xv: val data
        :param Yv: classes for val data
        :param verbose: print results
        :param alphas:
        :param lambdas:


        :return: best hyper params alpha and lambda
        """
        results = {}
        best_val = -1

        grid_search = [(lr, rg) for lr in alphas for rg in lambdas]
        for lr, rg in grid_search:
            # Create a new SVM instance
            for layer in self.layers:
                layer.alpha, layer.lambda_ = lr, rg
            train_loss, val_loss = self.train(X, y, batch=200)

            # Predict values for training set
            train_accuracy = metrics.accuracy(y, self.predict(X))
            val_accuracy = metrics.accuracy(Yv, self.predict(Xv))

            # Save results
            results[(lr, rg)] = (train_accuracy, val_accuracy)
            if best_val < val_accuracy:
                best_val = val_accuracy

        if verbose:
            for lr, reg in sorted(results):
                train_accuracy, val_accuracy = results[(lr, reg)]
                print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

        return max(results, key=lambda x: results[x])

    # def weight_mean_std(self):
    #     Hs = {}
    #     for  i in range()


class SVM(Regression):

    def __init__(self) -> None:
        super().__init__()
        # hyper params
        self.alpha: float = 1
        self.lambda_: float = 0
        self.c: float = 0
        self.gamma: float = 0
        # model params
        self.layers = []
        # engine params
        self.act: Activation() = None
        self.reg: Regularization() = None
        self.opt: Optimizer() = None

    def compile(self, alpha=0.001, lambda_=0., activation: Activation = Softmax(), reg: Regularization = L2(),
                opt: Optimizer = Vanilla, c=1, gamma=0) -> None:
        # super().compile()
        self.act, self.reg, self.opt = activation, reg, opt
        self.alpha, self.lambda_, self.c, self.gamma = alpha, lambda_, c, gamma

    def train(self, X: np.ndarray, y: np.ndarray, val: tuple = None, iter_=1500, batch=32, eps=0.001,
              return_loss=True, verbose=True) -> tuple[list, list]:
        Model.train(self, X, y)
        self.n, self.k = X.shape[1], np.max(y) + 1
        if len(self.layers) == 0:
            self.layers.append(Dense(self.k, input_shape=(self.n,), alpha=self.alpha, lambda_=self.lambda_,
                                     act=self.act, reg=self.reg, opt=self.opt, eps=eps))

        return super().train(X, y, val, iter_, batch, return_loss, verbose)
