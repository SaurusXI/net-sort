import numpy as np
from ..LSTM.cell import Cell


CONTEXT_LEN = 256


class Decoder:
    def __init__(self, output_len):
        self.cell = Cell()
        self.output_len = output_len

        # Initialize weights and biases
        self.weights = {
            'update': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'forget': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'candidate': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'output': np.random.random([CONTEXT_LEN, CONTEXT_LEN])
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
        self.contexts = None
        self.activations = None

        # Initialize stuff to store during backprop
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
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

    def forward(self, encoded_activation, encoded_context, timesteps):
        self.activations = np.zeros([CONTEXT_LEN, 1, timesteps])
        self.contexts = self.activations

        activation = encoded_activation
        context = encoded_context

        for t in range(timesteps):
            activation, context, cache = self.cell.forward(
                None, activation, context, self.weights, self.biases, False
            )
            self.activations[:, :, t] = activation
            self.contexts[:, :, t] = context
            self.caches.append(cache)

        self.input_activation = encoded_activation
        self.input_context = encoded_context
