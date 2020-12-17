import numpy as np


def OHE(val, len_output):
    val = np.array(val)
    res = np.eye(len_output)[np.array(val).reshape(-1)]
    return res.reshape(list(val.shape)+[len_output])


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    out = -np.sum(targets*np.log(predictions))/N
    return out
