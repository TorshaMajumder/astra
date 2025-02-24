#
# Import all dependencies
#
import numpy as np


def standardize(x, err):

    mean = np.average(x, weights=1/err**2)
    x_ = x - mean
    return x_