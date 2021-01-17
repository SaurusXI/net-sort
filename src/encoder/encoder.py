import numpy as np
from LSTM.cell import Cell
from model.utils import OHE


CONTEXT_LEN = 32


class Encoder:
    def __init__(self, input_len):
        self.cell = Cell()
        self.input_len = input_len
        # Initialize weights and biases
        self.weights = {
            'update': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len]) / 1e4,
            'forget': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len]) / 1e4,
            'candidate': np.random.random(
                [CONTEXT_LEN, CONTEXT_LEN + input_len]
            ) / 1e4,
            'output': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len]) / 1e4
        }
        # print(self.weights['update'][:10])
        # print(self.weights['forget'][:10])
        self.biases = {
            'update': np.random.random([CONTEXT_LEN, 1]),
            'forget': np.random.random([CONTEXT_LEN, 1]),
            'candidate': np.random.random([CONTEXT_LEN, 1]),
            'output': np.random.random([CONTEXT_LEN, 1])
        }

        # Initialize stuff to store during forward pass
        self.caches = []
        self.input = []
        self.a0 = np.random.random([CONTEXT_LEN, 1])
        self.contexts = None
        self.activations = None

        # Initialize stuff to store during backprop
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN + input_len]
        bias_shape = [CONTEXT_LEN, 1]
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
        }

    def forward(self, x):
        timesteps = x.shape[0]
        x = list(map(lambda i: OHE(i, self.input_len), x))
        self.activations = []
        self.contexts = []

        activation = self.a0
        context = np.random.random(activation.shape)

        for t in range(timesteps):
            # if t == 3:
            #     1/0
            activation, context, cache = self.cell.forward(
                x[t], activation, context, self.weights, self.biases
            )
            # print(activation[:10])
            self.activations.append(activation)
            self.contexts.append(context)
            self.caches.append(cache)

        self.input = x
        self.activations = np.array(self.activations)
        self.contexts = np.array(self.contexts)
        
        return self.activations, self.contexts

    def backprop(self, dactivation, dcontext):
        # print('backproping')
        timesteps = np.array(self.input).shape[0]

        for t in reversed(range(timesteps)):
            grad = self.cell.backprop(
                dactivation,
                dcontext,
                self.caches[t]
            )
            dactivation = grad['activ_prev']
            dcontext = grad['context_prev']
            self.update_grads(grad)

    def update_grads(self, grad, clipping=True):
        self.gradients['weights_forget'] += grad['weights_forget']
        self.gradients['weights_update'] += grad['weights_update']
        self.gradients['weights_output'] += grad['weights_output']
        self.gradients['weights_candidate'] += grad['weights_candidate']
        self.gradients['bias_forget'] += grad['bias_forget']
        self.gradients['bias_update'] += grad['bias_update']
        self.gradients['bias_output'] += grad['bias_output']
        self.gradients['bias_candidate'] += grad['bias_candidate']

        if clipping:
            self.clip_grads()

    def get_activations(self):
        return self.contexts

    def clip_grads(self):
        for k, v in self.gradients.items():
            np.clip(
                v, 1e-3, 1e3, out=self.gradients[k]
            )

    def reset_gradients(self):
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN + self.input_len]
        bias_shape = [CONTEXT_LEN, 1]
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
        }

    def apply_gradients(self, learning_rate=1e-3):
        self.weights['forget'] -= learning_rate * self.gradients['weights_forget']
        self.weights['update'] -= learning_rate * self.gradients['weights_update']
        self.weights['output'] -= learning_rate * self.gradients['weights_output']
        self.weights['candidate'] -= learning_rate * self.gradients['weights_candidate']
        self.biases['forget'] -= learning_rate * self.gradients['bias_forget']
        self.biases['update'] -= learning_rate * self.gradients['bias_update']
        self.biases['output'] -= learning_rate * self.gradients['bias_output']
        self.biases['candidate'] -= learning_rate * self.gradients['bias_candidate']
