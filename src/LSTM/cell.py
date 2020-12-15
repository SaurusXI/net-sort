import numpy as np
from scipy.special import expit


class Cell:
    def __init__(self):
        return

    def forward(x, activ_prev, context_prev, weights, biases):
        dim_x, m = x.shape
        X = np.concatenate([activ_prev, x])

        forget_gate = expit((weights['forget'] @ X) + biases['forget'])
        update_gate = expit((weights['update'] @ X) + biases['update'])
        out_gate = expit((weights['output'] @ X) + biases['output'])
        candidate = np.tanh((weights['candidate'] @ X) + biases['candidate'])
        context = (forget_gate * context_prev) + (update_gate * candidate)
        activations = out_gate * np.tanh(context)

        cache = {
            "activations": activations,
            "context": context,
            "activ_prev": activ_prev,
            "context_prev": context_prev,
            "forget_gate": forget_gate,
            "update_gate": update_gate,
            "out_gate": out_gate,
            "weights": weights,
            "biases": biases
        }

        return activations, context, cache
