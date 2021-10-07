import abc

import numpy as np
import metrics
import time
import os
import pickle
from typing import Callable, Union
from norm import Norm, Norm2
from activation import Activation, Softmax
from optimizer import Optimizer, Vanilla
from layer import Layer, WeightLayer, NormLayer, Dropout, Conv, ActLayer, EmbedLayer, ConnectEmbedWeightRNN, \
    VanillaRNN, ReshapeLayer
from data_generator import DataGen, DataGenRNN
import math


class Model:
    def __init__(self, classes: Union[list, np.ndarray] = None, mode: str = None) -> None:
        super().__init__()
        # general param
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        # hyper param
        self.threshold: float = 0.5
        # model param
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.classes: Union[list, np.ndarray] = classes
        # params
        self.path = None
        self.mode: str = mode

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

    def predict_class(self, X: np.ndarray, classes: list = None):
        if not self.classes and not classes:
            print('classes not specified')
            return
        elif not self.classes:
            self.classes = classes
        pred = self.predict(X)
        return classes[pred]

    def loss(self, X: np.ndarray, y: np.ndarray):
        pass

    def split(self):
        pass

    def save(self, file: str = None):
        pass

    def load(self):
        pass

    # def praper_data(self, X, y):
    #     X=
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


# class Regression(Model):
#
#     def __init__(self, classes: Union[list, np.ndarray] = None, mode: str = None) -> None:
#         super().__init__(classes, mode)
#         self.layers: list[Layer] = []
#
#     def compile(self) -> None:
#         layers = self.layers
#         assert layers is not None and len(layers) > 0
#
#         for i in range(1, len(layers)):
#             layers[i].input_shape = layers[i - 1].out_shape
#
#     def train(self, data: Union[DataGen, tuple[np.ndarray, np.ndarray]],
#               val: Union[DataGen, tuple[np.ndarray, np.ndarray], None] = None, batch=32,
#               epoch=100, return_loss=True, verbose=True, lrd=1., acc_func=metrics.accuracy) -> tuple[list, list]:
#         """
#
#         :param data: tuple of (X,y) or data generator
#         :param val: validation data (X_val,Y_val) or data generator
#         :param batch: batch size
#         :param epoch: number of epochs (every epoch iter over all data)
#         :param return_loss: return loss tuple: ([train loss],[val loss])
#         :param verbose: print every epoch the current loss & accuracy
#         :param lrd: float in range (0,1] for reduse the learning rate every epoch
#         :param acc_func: function to check the accuracy
#
#         :return: loss tuple: ([train loss],[val loss]) if return_loss=True, else ([],[])
#         """
#         if isinstance(data, tuple):
#             data = DataGen(data[0], data[1], shuffle=True, batch=batch)
#         data[0], data[1] = self.prepare_data(data[0], data[1])
#
#         if val:
#             if isinstance(val, tuple):
#                 val = DataGen(val[0], val[1], shuffle=False, batch=batch)
#             val[0], val[1] = self.prepare_data(val[0], val[1])
#
#             # unpacked param
#         layers = self.layers
#         batch = min(batch, len(data))
#         iter_ = max(1, len(data) // batch)
#         loss_history_t, loss_history_v = [], []
#         H_val, val_batch, x, y = None, None, None, None
#
#         for i in range(epoch):
#             for j, (x, y) in enumerate(data):
#
#                 if return_loss:
#                     loss_history_t.append(self.loss(x, y, mode='test'))
#                     if val:
#                         val_batch = next(val)
#                         H_val = self.feedforward(val_batch[0], mode='test')
#                         loss_history_v.append(self.loss(val_batch[0], val_batch[1], H_val[-1], mode='test'))
#                     if verbose:
#                         s = 'iteration %d / %d: loss %f acc %f' % (
#                             (i + 1) * j, epoch * iter_, loss_history_t[-1],
#                             acc_func(y, self.predict(x, self.feedforward(x, mode='test')[-1])))
#                         if val:
#                             s += ' val loss %f val acc %f' % \
#                                  (loss_history_v[-1], acc_func(val_batch[1], self.predict(val_batch[0], H_val[-1])))
#                         print(s)
#
#                 H = self.feedforward(x)
#                 self.backpropagation(H, y)
#                 time.sleep(7)
#
#             if lrd < 1.:
#                 for layer in layers:
#                     if isinstance(layer, (WeightLayer, NormLayer, Conv)):
#                         layer.alpha *= lrd
#
#         return loss_history_t, loss_history_v
#
#     def feedforward(self, X: np.ndarray, mode='train', prepare=False) -> list[np.ndarray]:
#         if prepare:
#             X = self.prepare_data(X)
#
#         layers, m = self.layers, X.shape[0]
#
#         H = [X]
#         for layer in layers:
#             H[-1] = H[-1].reshape((m, *layer.input_shape))
#             if isinstance(layer, (NormLayer, Dropout)):
#                 H.append(layer.forward(H[-1], mode=mode))
#             else:
#                 H.append(layer.forward(H[-1]))
#
#         return H
#
#     def backpropagation(self, H, y) -> None:
#         """
#         implement the chain roll for the gradient flow:
#             for each layer call his backward method
#
#         :param H: all the outputs from all layers [X,h_1,...,Prediction]
#         :param y: true labels for the data
#
#         """
#         layers, m = self.layers, H[0].shape[0]
#
#         delta = self.layers[-1].delta(y, H[-1])
#         for layer, h, h_next, i in zip(layers[:-1][::-1], H[:-2][::-1], H[1:-1][::-1], range(len(layers[:-1]))[::-1]):
#             delta = delta.reshape((m, *layer.out_shape))
#
#             if isinstance(layer, (NormLayer, Dropout)):
#                 delta = layer.backward(delta, h, return_delta=bool(i))
#             elif isinstance(layer, ActLayer):
#                 delta = layer.backward(delta, h, H=h_next, return_delta=bool(i))
#             else:
#                 delta = layer.backward(delta, h, return_delta=bool(i))
#
#     def grad(self, H: list[np.ndarray], y: np.ndarray) -> list[dict]:
#         layers, grades, m = self.layers, [], H[0].shape[0]
#
#         grades.append({'delta': self.layers[-1].delta(y, H[-1])})
#         for layer, h, i in zip(layers[:-1][::-1], H[:-2][::-1], range(len(layers[:-1][::-1]))):
#             grades[-1]['delta'] = grades[-1]['delta'].reshape((m, *layer.out_shape))
#             grades.append(layer.grad(grades[-1]['delta'], h))
#
#         return grades[::-1]
#
#     def loss(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray = None, mode='train', prepare: bool = False) -> float:
#         if pred is None:
#             pred = self.feedforward(X, mode=mode, prepare=prepare)[-1]  # , mode='train'
#         m, Loss, layers = pred.shape[0], self.layers[-1].loss, self.layers
#         J = Loss(y, pred)
#
#         for layer in layers:
#             if isinstance(layer, (Dense, WeightLayer, Conv)):
#                 J += layer.regularize() / 2
#         # J += sum(layer.regularize() if isinstance(layer, Dense) else 0 for layer in layers) / 2
#         # J += sum(layer.regularize() for layer in layers)
#         return J
#
#     def predict(self, X: np.ndarray, pred: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
#         if pred is None:
#             pred = self.feedforward(X, mode='test', prepare=True)[-1]
#         return np.argmax(pred, axis=1)
#
#     def predict_class(self, X: np.ndarray, pred: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
#         """
#         predict class label for every example in X
#
#         :param X: data to predict
#         :param pred: score for every class (=output of last layer)
#         :param threshold:
#
#         :return: predict class label for every example in X
#         """
#         if pred is None:
#             pred = self.feedforward(X, mode='test', prepare=True)[-1]
#         pred = np.argmax(pred, axis=1).reshape(-1)
#         return self.classes[pred]
#
#     def describe(self):
#         layers = self.layers
#         for i, layer in enumerate(layers):
#             print(f'{i}. {layer}')
#
#     def best_alpha_lambda(self, X, y, Xv, Yv, alphas, lambdas, verbose=True):
#         """
#         choose the best hyper params alpha and lambda for svm
#
#         :param X: train data
#         :param y: classes for train data
#         :param Xv: val data
#         :param Yv: classes for val data
#         :param verbose: print results
#         :param alphas:
#         :param lambdas:
#
#
#         :return: best hyper params alpha and lambda
#         """
#         results = {}
#         best_val = -1
#
#         grid_search = [(lr, rg) for lr in alphas for rg in lambdas]
#         for lr, rg in grid_search:
#             # Create a new SVM instance
#             for layer in self.layers:
#                 layer.alpha, layer.lambda_ = lr, rg
#             train_loss, val_loss = self.train(X, y, batch=200)
#
#             # Predict values for training set
#             train_accuracy = metrics.accuracy(y, self.predict(X))
#             val_accuracy = metrics.accuracy(Yv, self.predict(Xv))
#
#             # Save results
#             results[(lr, rg)] = (train_accuracy, val_accuracy)
#             if best_val < val_accuracy:
#                 best_val = val_accuracy
#
#         if verbose:
#             for lr, reg in sorted(results):
#                 train_accuracy, val_accuracy = results[(lr, reg)]
#                 print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
#
#         return max(results, key=lambda x: results[x])
#
#     def save(self, file: str = None) -> None:
#         if file:
#             self.path = os.path.join('/home/bb/Documents/python/MachineLearning/model data', file)
#
#         assert self.path is not None, 'file name not specified'
#
#         with open(self.path, 'wb') as f:
#             pickle.dump(self, f)
#
#     def load(self):
#         assert self.path is not None, 'file name not specified'
#
#         with open(self.path, 'rb') as f:
#             model: NN = pickle.load(f)
#         return model
#
#     def prepare_data(self, X: np.ndarray, y: np.ndarray = None, reg_func: Callable = None,
#                      rer_func_params: tuple = None) -> Union[tuple, np.ndarray]:
#
#         X = X.copy().astype(np.float64)
#         if self.mode.upper() == 'RGB':
#             X = X.transpose((0, 3, 1, 2))
#
#         if y is not None:
#             return X, y.reshape(-1)
#
#         return X
#
#     def __iadd__(self, other: Layer):
#         if isinstance(other, Dense):
#             other.input_shape = (self.layers[-1].out_shape,)
#
#         self.k = other.out_shape
#         self.layers.append(other)
#         return self


class KNearestNeighbor(Model):

    def __init__(self, reg: Norm) -> None:
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


class NN(Model):

    def __init__(self, layers: list[Layer] = None, classes: Union[list, np.ndarray] = None, mode: str = '') -> None:
        super().__init__(classes, mode)

        # model param
        self.layers: list[Layer] = layers
        # general params

    def compile(self) -> None:
        layers = self.layers
        assert layers is not None and len(layers) > 0

        for i in range(1, len(layers)):
            if not layers[i].input_shape:
                layers[i].input_shape = layers[i - 1].out_shape

    def train(self, data: Union[DataGen, tuple[np.ndarray, np.ndarray]],
              val: Union[DataGen, tuple[np.ndarray, np.ndarray], None] = None, batch=32,
              epoch=100, return_loss=True, verbose=True, lrd=1., acc_func=metrics.accuracy) -> tuple[list, list]:
        """

        :param data: tuple of (X,y) or data generator
        :param val: validation data (X_val,Y_val) or data generator
        :param batch: batch size
        :param epoch: number of epochs (every epoch iter over all data)
        :param return_loss: return loss tuple: ([train loss],[val loss])
        :param verbose: print every epoch the current loss & accuracy
        :param lrd: float in range (0,1] for reduse the learning rate every epoch
        :param acc_func: function to check the accuracy

        :return: loss tuple: ([train loss],[val loss]) if return_loss=True, else ([],[])
        """
        batch = min(batch, data[0].shape[0])
        if isinstance(data, tuple):
            data = DataGen(data[0], data[1], shuffle=True, batch=batch)
        data[0], data[1] = self.prepare_data(data[0], data[1])

        if val:
            if isinstance(val, tuple):
                val = DataGen(val[0], val[1], shuffle=False, batch=batch)
            val[0], val[1] = self.prepare_data(val[0], val[1])

            # unpacked param
        layers = self.layers
        iter_ = max(1, len(data) // batch)
        loss_history_t, loss_history_v = [], []
        H_val, val_batch, x, y = None, None, None, None

        for i in range(epoch):
            for j, (x, y) in enumerate(data):

                if return_loss:
                    loss_history_t.append(self.loss(x, y, mode='test'))
                    if val:
                        val_batch = next(val)
                        H_val = self.feedforward(val_batch[0], mode='test')
                        loss_history_v.append(self.loss(val_batch[0], val_batch[1], H_val[-1], mode='test'))
                    if verbose:
                        s = 'iteration %d / %d: loss %f acc %f' % (
                            (i + 1) * j, epoch * iter_, loss_history_t[-1],
                            acc_func(y, self.predict(x, self.feedforward(x, mode='test')[-1])))
                        if val:
                            s += ' val loss %f val acc %f' % \
                                 (loss_history_v[-1], acc_func(val_batch[1], self.predict(val_batch[0], H_val[-1])))
                        print(s)

                H = self.feedforward(x)
                self.backpropagation(H, y)
                time.sleep(1)

            if lrd < 1.:
                for layer in layers:
                    if isinstance(layer, (WeightLayer, NormLayer, Conv)):
                        layer.alpha *= lrd

        return loss_history_t, loss_history_v

    def feedforward(self, X: np.ndarray, mode='train', prepare=False) -> list[np.ndarray]:
        if prepare:
            X = self.prepare_data(X)

        layers = self.layers

        H = [X]
        for layer in layers:
            if isinstance(layer, (NormLayer, Dropout)):
                H.append(layer.forward(H[-1], mode=mode))
            else:
                H.append(layer.forward(H[-1]))

        return H

    def backpropagation(self, H, y) -> None:
        """
        implement the chain roll for the gradient flow:
            for each layer call his backward method

        :param H: all the outputs from all layers [X,h_1,...,Prediction]
        :param y: true labels for the data

        """
        layers = self.layers

        delta = self.layers[-1].delta(y, H[-1])
        for layer, h, h_next, i in zip(layers[:-1][::-1], H[:-2][::-1], H[1:-1][::-1], range(len(layers[:-1]))[::-1]):
            if isinstance(layer, (NormLayer, Dropout)):
                delta = layer.backward(delta, h, return_delta=bool(i))
            elif isinstance(layer, ActLayer):
                delta = layer.backward(delta, h, H=h_next, return_delta=bool(i))
            else:
                delta = layer.backward(delta, h, return_delta=bool(i))

    def grad(self, H: list[np.ndarray], y: np.ndarray) -> list[dict]:
        layers, grades = self.layers, []

        grades.append({'delta': self.layers[-1].delta(y, H[-1])})
        for layer, h, h_next, i in zip(layers[:-1][::-1], H[:-2][::-1], H[1:-1][::-1], range(len(layers[:-1]))[::-1]):
            # delta = delta.reshape((m, *layer.out_shape))

            delta = grades[-1]['delta']
            if isinstance(layer, (NormLayer, Dropout)):
                grades.append(layer.grad(delta, h))
            elif isinstance(layer, ActLayer):
                grades.append(layer.grad(delta, h))
            else:
                grades.append(layer.grad(delta, h))

        return grades[::-1]

    def loss(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray = None, mode='train', prepare: bool = False,
             data=None) -> float:
        if pred is None:
            pred = self.feedforward(X, mode=mode, prepare=prepare)[-1]  # , mode='train'

        Loss, layers = self.layers[-1].loss, self.layers
        J = Loss(y, pred)

        for layer in layers:
            if isinstance(layer, (WeightLayer, Conv)):
                J += layer.regularize() / 2
        # J += sum(layer.regularize() if isinstance(layer, Dense) else 0 for layer in layers) / 2
        # J += sum(layer.regularize() for layer in layers)
        return J

    def predict(self, X: np.ndarray, pred: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
        if pred is None:
            pred = self.feedforward(X, mode='test', prepare=True)[-1]
        return np.argmax(pred, axis=1)

    def predict_class(self, X: np.ndarray, pred: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
        """
        predict class label for every example in X

        :param X: data to predict
        :param pred: score for every class (=output of last layer)
        :param threshold:

        :return: predict class label for every example in X
        """
        if pred is None:
            pred = self.feedforward(X, mode='test', prepare=True)[-1]
        pred = np.argmax(pred, axis=1).reshape(-1)
        return self.classes[pred]

    def describe(self):
        layers = self.layers
        for i, layer in enumerate(layers):
            print(f'{i}. {layer}')

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

    def save(self, file: str = None) -> None:
        if file:
            self.path = os.path.join('/home/bb/Documents/python/MachineLearning/model data', file)

        assert self.path is not None, 'file name not specified'

        with open(self.path, 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        assert self.path is not None, 'file name not specified'

        with open(self.path, 'rb') as f:
            model: NN = pickle.load(f)
        return model

    def prepare_data(self, X: np.ndarray, y: np.ndarray = None, reg_func: Callable = None,
                     rer_func_params: tuple = None) -> Union[tuple, np.ndarray]:

        X = X.copy().astype(np.float64)
        if self.mode.upper() == 'RGB':
            X = X.transpose((0, 3, 1, 2))

        if y is not None:
            return X, y.reshape(-1)

        return X

    def __iadd__(self, other: Layer):
        self.layers.append(other)
        return self


class RNN(NN):

    def __init__(self, data_dic: dict, layers: list[Layer] = None, mode: str = '') -> None:
        super().__init__(layers, None, mode)
        # self.caption: np.ndarray = np.array([])
        self.data_dic: dict = data_dic
        self._null = data_dic["<NULL>"]
        self._start = data_dic.get("<START>", None)
        self._end = data_dic.get("<END>", None)
        self.layers: list[Layer] = layers

        self.connected_idx = 0
        self.rnn_idx_start = 1
        self.rnn_idx_end = 2
        self.act_idx = 3

    def compile(self) -> None:
        super().compile()

    def train(self, data: Union[DataGenRNN, tuple],
              val: Union[DataGenRNN, tuple, None] = None, batch=32,
              epoch=100, return_loss=True, verbose=True, lrd=1., acc_func=metrics.accuracy) -> tuple[list, list]:
        """

        :param data: tuple of (X,y) or data generator
        :param val: validation data (X_val,Y_val) or data generator
        :param batch: batch size
        :param epoch: number of epochs (every epoch iter over all data)
        :param return_loss: return loss tuple: ([train loss],[val loss])
        :param verbose: print every epoch the current loss & accuracy
        :param lrd: float in range (0,1] for reduse the learning rate every epoch
        :param acc_func: function to check the accuracy

        :return: loss tuple: ([train loss],[val loss]) if return_loss=True, else ([],[])
        """

        batch = min(batch, data[0].shape[0])
        if isinstance(data, tuple):
            caption_in = data[0][:, :-1]
            caption_out = data[0][:, 1:]
            mask = np.array(caption_out != self._null)
            data = DataGenRNN([caption_in, data[1]], caption_out, mask, shuffle=True, batch=batch)

        if val:
            if isinstance(val, tuple):
                caption_in = val[0][:, :-1]
                caption_out = val[0][:, 1:]
                mask = np.array(caption_out != self._null)
                val = DataGenRNN([caption_in, val[1]], caption_out, mask, shuffle=False, batch=batch)

        layers = self.layers
        iter_ = max(1, math.ceil(len(data) / batch))
        loss_history_t, loss_history_v = [], []
        H_val, x_val, y_val, mask_val = None, None, None, None

        for i in range(epoch):
            for j, (x, y, mask) in enumerate(data):

                if return_loss:
                    self.layers[-1].data['mask'] = mask
                    loss_history_t.append(self.loss(x, y, mode='test'))
                    if val:
                        x_val, y_val, mask_val = next(val)
                        H_val = self.feedforward(x_val, mode='test')
                        self.layers[-1].data['mask'] = mask_val
                        loss_history_v.append(self.loss(x_val, y_val, H_val[-1], mode='test'))
                    if verbose:
                        s = 'iteration %d / %d: loss %f acc %f' % (
                            i * epoch + j + 1, epoch * iter_, loss_history_t[-1],
                            acc_func(y, self.predict(x, self.feedforward(x, mode='test')[-1], y.shape)))
                        if val:
                            s += ' val loss %f val acc %f' % \
                                 (loss_history_v[-1], acc_func(y_val, self.predict(x_val, H_val[-1], y_val.shape)))
                        print(s)
                self.layers[-1].data['mask'] = mask
                H = self.feedforward(x)
                self.backpropagation(H, y)
                time.sleep(1)

            if lrd < 1.:
                for layer in layers:
                    if isinstance(layer, (WeightLayer, NormLayer, Conv, VanillaRNN)):
                        layer.alpha *= lrd

        return loss_history_t, loss_history_v

    def predict(self, X: Union[np.ndarray, tuple], pred: np.ndarray = None, shape=None,
                threshold: float = 0.5) -> np.ndarray:
        if pred is None:
            pred = self.feedforward(X, mode='test', prepare=True)[-1]
        if shape:
            pred = np.reshape(pred, shape + (-1,))
        return np.argmax(pred, axis=2)

    def sample(self, features, max_steps=30) -> np.ndarray:
        m = features.shape[0]
        x0 = (np.array([self._start] * m))
        captions = self._null * np.ones((m, max_steps), dtype=np.int32)

        data = x0[:, None], features
        shape = (m, 1)
        for i in range(max_steps):
            x = captions[:, i] = self.predict(data, shape=shape).reshape(-1)
            data = (x[:, None],)
        return captions

    def prepare_data(self, X: np.ndarray, y: np.ndarray = None, reg_func: Callable = None,
                     rer_func_params: tuple = None) -> Union[tuple, np.ndarray]:
        return X

# class SVM(NN):

#
#     def __init__(self) -> None:
#         super().__init__()
#         # hyper params
#         self.alpha: float = 1
#         self.lambda_: float = 0
#         self.c: float = 0
#         self.gamma: float = 0
#         # model params
#         self.layers = []
#         # engine params
#         self.act: Activation() = None
#         self.reg: Regularization() = None
#         self.opt: Optimizer() = None
#
#     def compile(self, alpha=0.001, lambda_=0., activation: Activation = Softmax(), reg: Regularization = L2(),
#                 opt: Optimizer = Vanilla, c=1, gamma=0) -> None:
#         # super().compile()
#         self.act, self.reg, self.opt = activation, reg, opt
#         self.alpha, self.lambda_, self.c, self.gamma = alpha, lambda_, c, gamma
#
#     def train(self, X: np.ndarray, y: np.ndarray, val: tuple = None, iter_=1500, batch=32, eps=0.001,
#               return_loss=True, verbose=True) -> tuple[list, list]:
#         Model.train(self, X, y)
#         self.n, self.k = X.shape[1], np.max(y) + 1
#         if len(self.layers) == 0:
#             self.layers.append(Dense(self.k, input_shape=(self.n,), alpha=self.alpha, lambda_=self.lambda_,
#                                      act=self.act, reg=self.reg, opt=self.opt, eps=eps))
#
#         return super().train(X, y, val, iter_, batch, return_loss, verbose)
