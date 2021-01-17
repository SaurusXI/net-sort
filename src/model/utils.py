import numpy as np
from scipy.stats import entropy


def OHE(val, len_output):
    val = np.array(val)
    res = np.eye(len_output)[np.array(val).reshape(-1)]
    return res.reshape(list(val.shape)+[len_output])


def cross_entropy(predictions, targets):
    predictions = np.array(predictions)[:, :, 0]
    # print(predictions.shape)
    # print(targets.shape)
    return entropy(predictions) + entropy(predictions, targets)


def relu(x):
    x = list(map(lambda i: i if i > 0 else 0, x))
    return np.array(x)


def categorical_cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    predictions = predictions[:, :, 0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def dsoftmax(s):
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
