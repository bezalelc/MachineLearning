from typing import Union, List
import pickle
import numpy as np
import os

"""
CIFAR10 url => https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
"""
data_paths = {'CIFAR10': '/home/bb/Documents/python/Algorithms/Algorithms/ML_/data/cifar-10-batches-py'}


def CIFAR_10_batch(filename) -> tuple[np.ndarray, np.ndarray]:
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def CIFAR_10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ load all of cifar """
    path = data_paths['CIFAR10']
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b,))
        X, Y = CIFAR_10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = CIFAR_10_batch(os.path.join(path, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
