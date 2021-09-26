import numpy as np
from typing import NewType, Callable
from numpy.random import randn

InitWeights = NewType('InitWeights', Callable[[tuple, float], np.ndarray])
stdScale = InitWeights(lambda shape, std: randn(*shape) * std)
xavierScale = InitWeights(lambda shape, std: randn(*shape) / shape[0] ** 0.5)
heEtAlScale = InitWeights(lambda shape, std: randn(*shape) / (shape[0] / 2.) ** 0.5)
