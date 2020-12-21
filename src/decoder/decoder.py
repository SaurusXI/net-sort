import numpy as np
from LSTM.cell import Cell
from model.utils import relu


CONTEXT_LEN = 256


class Decoder:
    def __init__(self, output_len):
        self.cell = Cell()
        self.output_len = output_len

        # Initialize weights and biases
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN + self.output_len]
        bias_shape = [CONTEXT_LEN, 1]
        self.weights = {
            'update': np.random.random(weights_shape) / 1e4,
            'forget': np.random.random(weights_shape) / 1e4,
            'candidate': np.random.random(weights_shape) / 1e4,
            'output': np.random.random(weights_shape) / 1e4,
            'y': np.random.random([1, CONTEXT_LEN])
        }
        self.biases = {
            'update': np.random.random([CONTEXT_LEN, 1]),
            'forget': np.random.random([CONTEXT_LEN, 1]),
            'candidate': np.random.random([CONTEXT_LEN, 1]),
            'output': np.random.random([CONTEXT_LEN, 1]),
            'y': 0
        }

        # Initialize stuff to store during forward pass
        self.caches = []
        self.input = []
        self.contexts = None
        self.activations = None
        self.predictions = []

        # Initialize stuff to store during backprop
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_y': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': 0,
            'output_activation': np.zeros(bias_shape),
            'encoder_activations': []
        }

    def forward(self, encoded_activations, encoded_contexts, timesteps):
        self.activations = np.zeros([CONTEXT_LEN, 1, timesteps])
        self.contexts = self.activations
        self.timesteps = timesteps
        self.predictions = []
        self.gradients['encoder_activations'] = []
        prediction = -1
        context = encoded_contexts[:, :, -1]

        for t in range(timesteps):
            activation = encoded_activations[:, :, t]
            # print(activation)
            activation, context, cache = self.cell.forward(
                np.array(prediction), activation, context, self.weights, self.biases
            )
            # print(activation)
            prediction = relu((self.weights['y'] @ activation)[0][0]
                              + self.biases['y'])
            self.predictions.append(prediction)
            self.activations[:, :, t] = activation
            self.contexts[:, :, t] = context
            self.caches.append(cache)

        self.input_activation = encoded_activations
        self.input_context = encoded_contexts

        return self.predictions

    def backprop(self, ground_truth):
        dcontext = np.zeros([CONTEXT_LEN, 1])

        for t in reversed(range(self.timesteps)):
            zi = ((self.weights['y'] @ self.activations[:, :, t])
                  + self.biases['y'])[0][0]
            drelu = 1 if zi > 0 else 0
            doi = -(ground_truth[t] / self.predictions[t])

            self.gradients['weights_y'] = doi * drelu * self.activations[:, :, t].T
            self.gradients['bias_y'] += doi * drelu
            self.gradients['output_activation'] = doi * drelu * \
                self.weights['y'].T

            grad = self.cell.backprop(
                self.gradients['output_activation'],
                dcontext,
                self.caches[t]
            )
            dcontext = grad['context_prev']
            self.update_grads(grad)

        self.gradients['encoder_activations'] = np.array(
            self.gradients['encoder_activations']
        )
        return self.gradients['encoder_activations'], dcontext

    def update_grads(self, grad, clipping=True):
        self.gradients['weights_forget'] += grad['weights_forget']
        self.gradients['weights_update'] += grad['weights_update']
        self.gradients['weights_output'] += grad['weights_output']
        self.gradients['weights_candidate'] += grad['weights_candidate']
        self.gradients['bias_forget'] += grad['bias_forget']
        self.gradients['bias_update'] += grad['bias_update']
        self.gradients['bias_output'] += grad['bias_output']
        self.gradients['bias_candidate'] += grad['bias_candidate']
        self.gradients['encoder_activations'].append(
            grad['activ_prev']
        )
        if clipping:
            self.clip_grads()

    def reset_gradients(self):
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN + self.output_len]
        bias_shape = [CONTEXT_LEN, 1]
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_y': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': 0,
            'output_activation': np.zeros(bias_shape),
            'encoder_activations': []
        }

    def get_activations(self):
        return self.contexts

    def clip_grads(self):
        try:
            for k, v in self.gradients.items():
                np.clip(
                    v, 1e-3, 1e3, out=self.gradients[k]
                )
        except Exception:
            pass

    def apply_gradients(self, learning_rate=1e-3):
        # print(f'orig {self.weights["forget"][-1][0]}')
        # print(f'grad {self.gradients["weights_forget"][-1][0]}')
        self.weights['forget'] -= learning_rate * \
            self.gradients['weights_forget']
        # print(f'new {self.weights["forget"][-1][0]}')
        self.weights['update'] -= learning_rate * \
            self.gradients['weights_update']
        self.weights['output'] -= learning_rate * \
            self.gradients['weights_output']
        self.weights['candidate'] -= learning_rate * \
            self.gradients['weights_candidate']
        self.weights['y'] -= learning_rate * self.gradients['weights_y']
        self.biases['forget'] -= learning_rate * self.gradients['bias_forget']
        self.biases['update'] -= learning_rate * self.gradients['bias_update']
        self.biases['output'] -= learning_rate * self.gradients['bias_output']
        self.biases['candidate'] -= learning_rate * \
            self.gradients['bias_candidate']
        self.biases['y'] -= learning_rate * self.gradients['bias_y']
