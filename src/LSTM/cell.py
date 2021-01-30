import numpy as np
from scipy.special import expit


class Cell:
    def __init__(self):
        return

    def forward(
            self,
            x,
            activ_prev,
            context_prev,
            weights,
            biases,
            take_input=True):
        '''
        Forward pass through LSTM cell
        '''

        if take_input:
            X = np.concatenate([activ_prev, x.reshape((-1, 1))], axis=0)
        else:
            x = activ_prev
            X = x

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

    def backprop(self, dactivation, dcontext, cache, take_input=True):
        '''
        Backward pass through LSTM cell to compute gradients
        '''
        context = cache['context']
        activ_prev = cache['activ_prev']
        context_prev = cache['context_prev']
        forget_gate = cache['forget_gate']
        update_gate = cache['update_gate']
        out_gate = cache['out_gate']
        candidate = cache['candidate']
        x = cache['input']
        weights = cache['weights']

        out_len = dactivation.shape[0]

        dcontext += dactivation * out_gate * (1 - np.square(np.tanh(context)))
        dout_gate = dactivation * np.tanh(context)
        dforget_gate = dcontext * context_prev
        dupdate_gate = dcontext * candidate
        dcandidate = dcontext * update_gate
        dcontext_prev = dcontext * forget_gate

        if take_input:
            X = np.concatenate([activ_prev, x.reshape((-1, 1))], axis=0)
        else:
            X = activ_prev

        db_candidate = dcandidate * (1 - np.square(candidate))
        dW_candidate = db_candidate @ X.T
        db_out = dout_gate * out_gate * (1 - out_gate)
        dW_out = db_out @ X.T
        db_update = dupdate_gate * update_gate * (1 - update_gate)
        dW_update = db_update @ X.T
        db_forget = dforget_gate * forget_gate * (1 - forget_gate)
        dW_forget = db_forget @ X.T

        # Gradients for weights and biases
        dX_forget = weights['forget'].T @ db_forget
        dX_update = weights['update'].T @ db_update
        dX_out = weights['output'].T @ db_out
        dX_candidate = weights['candidate'].T @ db_candidate
        dX = dX_forget + dX_update + dX_out + dX_candidate

        if take_input:
            dactiv_prev = dX[:out_len, :]
            dx = dX[out_len:, :]
        else:
            dactiv_prev = dX
            dx = dX

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

        gradients = clip_gradients(gradients)
        return gradients


def clip_gradients(grads):
    for k, v in grads.items():
        np.clip(
            v, 1e-3, 1e3, out=grads[k]
        )
    return grads
