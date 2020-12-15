import numpy as np
from scipy.special import expit


class Cell:
    def __init__(self):
        return

    def forward(self, x, activ_prev, context_prev, weights, biases):
        X = np.concatenate([activ_prev, x], axis=0)

        forget_gate = expit((weights['forget'] @ X) + biases['forget'])
        update_gate = expit((weights['update'] @ X) + biases['update'])
        out_gate = expit((weights['output'] @ X) + biases['output'])
        candidate = np.tanh((weights['candidate'] @ X) + biases['candidate'])
        context = (forget_gate * context_prev) + (update_gate * candidate)
        activation = out_gate * np.tanh(context)

        cache = {
            "activation": activation,
            "context": context,
            "activ_prev": activ_prev,
            "context_prev": context_prev,
            "forget_gate": forget_gate,
            "update_gate": update_gate,
            "candidate": candidate,
            "out_gate": out_gate,
            "input": x,
            "weights": weights,
            "biases": biases
        }

        return activation, context, cache

    def backprop(self, dactivation, cache):
        activation = cache['activation']
        context = cache['context']
        activ_prev = cache['activ_prev']
        context_prev = cache['context_prev']
        forget_gate = cache['forget_gate']
        update_gate = cache['update_gate']
        out_gate = cache['out_gate']
        candidate = cache['candidate']
        x = cache['input']
        weights = cache['weights']

        out_len = activation.shape[0]

        # Gradients for LSTM cell gates
        dcontext = dactivation * out_gate * (1 - np.square(np.tanh(context)))
        dout_gate = dactivation * np.tanh(context) * out_gate * (1 - out_gate)

        dupdate_gate = (dcontext * candidate +
                        out_gate * (1 - np.square(np.tanh(context))) *
                        candidate * dactivation) * update_gate * \
            (1 - update_gate)

        dforget_gate = (dcontext * context_prev +
                        out_gate * (1 - np.square(np.tanh(context))) *
                        context_prev * dactivation) * forget_gate * \
            (1 - forget_gate)

        dcandidate = (dcontext * update_gate +
                      out_gate * (1 - np.square(np.tanh(context))) *
                      update_gate * dactivation) * \
            (1 - np.square(candidate))

        X = np.concatenate([activ_prev, x], axis=0)

        # Gradients for weights and biases
        dW_out = dout_gate @ X.T
        db_out = np.sum(dout_gate, axis=1, keepdims=True)
        dW_update = dupdate_gate @ X.T
        db_update = np.sum(dupdate_gate, axis=1, keepdims=True)
        dW_forget = dforget_gate @ X.T
        db_forget = np.sum(dforget_gate, axis=1, keepdims=True)
        dW_candidate = dcandidate @ X.T
        db_candidate = np.sum(dcandidate, axis=1, keepdims=True)

        # Gradients for previous time-step activation
        dactiv_prev = (weights['forget'][:, :out_len].T @ dforget_gate) + \
            (weights['update'][:, :out_len].T @ dupdate_gate) + \
            (weights['candidate'][:, :out_len].T @ dcandidate) + \
            (weights['output'][:, :out_len].T @ dout_gate)

        dcontext_prev = dcontext*forget_gate + \
            out_gate*(1 - np.square(np.tanh(context))) * forget_gate * \
            dactivation

        dx = (weights['forget'][:, out_len:].T @ dforget_gate) +\
            (weights['update'][:, out_len:].T @ dupdate_gate) +\
            (weights['candidate'][:, out_len:].T @ dcandidate) +\
            (weights['output'][:, out_len:].T @ dout_gate)

        # Gradients compiled into single dict
        gradients = {
            'input': dx,
            'activ_prev': dactiv_prev,
            'context_prev': dcontext_prev,
            'weights_forget': dW_forget,
            'weights_update': dW_update,
            'weights_output': dW_out,
            'weights_candidate': dW_candidate,
            'bias_forget': db_forget,
            'bias_update': db_update,
            'bias_output': db_out,
            'bias_candidate': db_candidate,
        }

        return gradients
