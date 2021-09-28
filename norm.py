import numpy as np


def standard_deviation(data, ddof=0):
    """
    Compute the standard deviation of data

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        mu: numpy array of the mean of every column (=attribute)
        sigma: numpy array of standard deviation of every column (=attribute)
    """
    '''
    m, n = data.shape
    mu = 1 / m * np.sum(data, axis=0)
    sigma = np.sqrt(1 / m * np.sum((data - mu) ** 2, axis=0))
    sigma[sigma == 0] = 1  // if sigma[i] == 0 => need to divide by 1 because there is not standard deviation  
    '''
    mu, sigma = np.mean(data, axis=0), np.std(data, axis=0, ddof=ddof)
    sigma[sigma == 0] = 1
    data = (data - mu) / sigma
    return data, mu, sigma


def simple_normalize(data):
    """
    Compute data between [-1,1]

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        max_: numpy array of max in every column (=attribute)
        min_: numpy array of min in every column (=attribute)
    """
    max_, min_ = np.max(data, axis=0), np.min(data, axis=0)
    div = (max_ - min_)
    div[div == 0] = 1
    data = (data - min_) / div
    return data, max_, min_


def sub_mean(data, axis: int = 0):
    return data - np.mean(data, axis=axis)
