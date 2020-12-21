import numpy as np
from LSTM.cell import Cell


CONTEXT_LEN = 256


class Encoder:
    def __init__(self, input_len):
        self.cell = Cell()
        # Initialize weights and biases
        self.weights = {
            'update': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len]),
            'forget': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len]),
            'candidate': np.random.random(
                [CONTEXT_LEN, CONTEXT_LEN + input_len]
            ),
            'output': np.random.random([CONTEXT_LEN, CONTEXT_LEN + input_len])
        }
        self.biases = {
            'update': np.random.random([CONTEXT_LEN, 1]),
            'forget': np.random.random([CONTEXT_LEN, 1]),
            'candidate': np.random.random([CONTEXT_LEN, 1]),
            'output': np.random.random([CONTEXT_LEN, 1])
        }

        # Initialize stuff to store during forward pass
        self.caches = []
        self.input = []
        self.a0 = np.zeros([CONTEXT_LEN, 1])
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
        self.activations = np.zeros([CONTEXT_LEN, 1, timesteps])
        self.contexts = self.activations

        activation = self.a0
        context = np.zeros(activation.shape)

        for t in range(timesteps):
            activation, context, cache = self.cell.forward(
                x[t], activation, context, self.weights, self.biases
            )
            # print(f'ctx{context.shape}')
            self.activations[:, :, t] = activation
            self.contexts[:, :, t] = context
            self.caches.append(cache)

        self.input = x
        output_activation = self.activations[:, :, timesteps-1]
        output_context = self.contexts[:, :, timesteps-1]

        return output_activation, output_context

    def backprop(self, dactivation, dcontexts):
        # print(dcontexts.shape)
        timesteps = self.input.shape[0]

        for t in reversed(range(timesteps)):
            grad = self.cell.backprop(
                dactivation,
                dcontexts[t, :, :],
                self.caches[t]
            )
            dactivation = grad['activ_prev']
            self.update_grads(grad)

    def update_grads(self, grad):
        self.gradients['weights_forget'] += grad['weights_forget']
        self.gradients['weights_update'] += grad['weights_update']
        self.gradients['weights_output'] += grad['weights_output']
        self.gradients['weights_candidate'] += grad['weights_candidate']
        self.gradients['bias_forget'] += grad['bias_forget']
        self.gradients['bias_update'] += grad['bias_update']
        self.gradients['bias_output'] += grad['bias_output']
        self.gradients['bias_candidate'] += grad['bias_candidate']

    def get_activations(self):
        return self.contexts

    def apply_gradients(self, learning_rate=1e-3):
        self.weights['forget'] -= learning_rate * self.gradients['weights_forget']
        self.weights['update'] -= learning_rate * self.gradients['weights_update']
        self.weights['output'] -= learning_rate * self.gradients['weights_output']
        self.weights['candidate'] -= learning_rate * self.gradients['weights_candidate']
        self.biases['forget'] -= learning_rate * self.gradients['bias_forget']
        self.biases['update'] -= learning_rate * self.gradients['bias_update']
        self.biases['output'] -= learning_rate * self.gradients['bias_output']
        self.biases['candidate'] -= learning_rate * self.gradients['bias_candidate']
