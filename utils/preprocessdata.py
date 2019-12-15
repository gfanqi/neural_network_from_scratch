import numpy as np


def preprocessdata(X, y):
    one = np.ones((len(X), 1))
    X = np.hstack((X, one))
    y = y.reshape((-1,1))
    return X, y
