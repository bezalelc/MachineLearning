import numpy as np


class DataGen:
    """
    class tha represent data generator for iter over large amount of data
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, shuffle=False, batch=32) -> None:
        """
        :param X: data in shape (number of training examples X number fo attributes X dimension...)
        :param y: labels of the data
        :param shuffle: shuffle the data
        :param batch: batch size
        """
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.shuffle: bool = shuffle
        self.batch: int = batch

    def __iter__(self):
        m, b = self.X.shape[0], self.batch
        idx = np.arange(m)
        if self.shuffle:
            np.random.shuffle(idx)
        return iter((self.X[idx[i:i + b]], self.y[idx[i:i + b]]) for i in range(0, m, b))

    def __next__(self):
        """
        :return: random sample of the data
        """
        batch_idx = np.random.choice(self.X.shape[0], self.batch, replace=False)
        return self.X[batch_idx], self.y[batch_idx]

    def __getitem__(self, key):
        if key == 0:
            return self.X
        elif key == 1:
            return self.y
        else:
            raise ValueError('key is 0 or 1: DataGen[0]=X,DataGen[1]=y')

    def __setitem__(self, key, value: np.ndarray):
        if key == 0:
            self.X = value
        elif key == 1:
            self.y = value
        else:
            raise ValueError('key is 0 or 1: DataGen[0]=X,DataGen[1]=y')

    def __len__(self):
        return self.X.shape[0]
