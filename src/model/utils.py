import numpy as np
from scipy.stats import entropy


def OHE(val, len_output):
    val = np.array(val)
    res = np.eye(len_output)[np.array(val).reshape(-1)]
    return res.reshape(list(val.shape)+[len_output])


def cross_entropy(predictions, targets, epsilon=1e-12):
    return entropy(targets) + entropy(targets, predictions)


def relu(x):
    return x if x > 0 else 0


def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
