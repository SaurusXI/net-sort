import numpy as np
from LSTM.cell import Cell
from model.utils import relu, OHE, drelu
from scipy.special import softmax


CONTEXT_LEN = 64


class Decoder:
    def __init__(self, output_len, temperature):
        self.cell = Cell()
        self.temperature = temperature
        self.output_len = output_len

        # Initialize weights and biases
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
        bias_shape = [CONTEXT_LEN, 1]
        self.weights = {
            'update': np.random.random(weights_shape) / 1e4,
            'forget': np.random.random(weights_shape) / 1e4,
            'candidate': np.random.random(weights_shape) / 1e4,
            'output': np.random.random(weights_shape) / 1e4,
            'y': np.random.random([self.output_len, CONTEXT_LEN]) / 1e4
        }
        # print(self.weights['y'])
        self.biases = {
            'update': np.random.random([CONTEXT_LEN, 1]),
            'forget': np.ones([CONTEXT_LEN, 1]),
            'candidate': np.random.random([CONTEXT_LEN, 1]),
            'output': np.random.random([CONTEXT_LEN, 1]),
            'y': np.random.random([self.output_len, 1])
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
            'weights_y': np.zeros([self.output_len, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1]),
            'output_activation': np.zeros(bias_shape),
        }

    def forward(self, encoded_activations, encoded_contexts, timesteps, debug=False):
        self.activations = []
        self.contexts = []
        self.timesteps = timesteps
        self.predictions = []
        prediction = np.zeros([1, self.output_len])
        context = encoded_contexts[-1]
        activation = encoded_activations[-1]

        for t in range(timesteps):
            # if t == 3:
            #     1/0
            activation, context, cache = self.cell.forward(
                None, activation, context, self.weights, self.biases, False
            )
            prediction = softmax(
                ((self.weights['y'] @ activation) + self.biases['y']) 
                / self.temperature
            )
            # print(prediction[:60])
            self.predictions.append(prediction)
            self.activations.append(activation)
            self.contexts.append(context)
            self.caches.append(cache)

        self.activations = np.array(self.activations)
        self.contexts = np.array(self.contexts)

        self.input_activation = encoded_activations
        self.input_context = encoded_contexts

        if debug:
            with open('activations.log', 'w') as f:
                print(np.array(self.predictions), file=f)
        return self.predictions

    def backprop(self, ground_truth):
        dcontext = np.zeros([CONTEXT_LEN, 1])
        dactiv_prev = np.zeros([CONTEXT_LEN, 1])

        for t in reversed(range(self.timesteps)):
            do = self.predictions[t] - OHE(ground_truth[t], self.output_len).reshape(-1, 1)
            self.gradients['output_activation'] = dactiv_prev + ((self.weights['y'].T @ do) / self.temperature)

            grad = self.cell.backprop(
                self.gradients['output_activation'],
                dcontext,
                self.caches[t],
                False
            )
            dcontext = grad['context_prev']
            dactiv_prev = grad['activ_prev']

            grad['weights_y'] = (do @ self.activations[t].T) / self.temperature
            grad['bias_y'] = do / self.temperature
            self.update_grads(grad)

        return dactiv_prev, dcontext

    def update_grads(self, grad, clipping=True):
        # print(self.gradients['weights_forget'].shape)
        # print(grad['weights_forget'].shape)
        self.gradients['weights_forget'] += grad['weights_forget']
        self.gradients['weights_update'] += grad['weights_update']
        self.gradients['weights_output'] += grad['weights_output']
        self.gradients['weights_candidate'] += grad['weights_candidate']
        self.gradients['bias_forget'] += grad['bias_forget']
        self.gradients['bias_update'] += grad['bias_update']
        self.gradients['bias_output'] += grad['bias_output']
        self.gradients['bias_candidate'] += grad['bias_candidate']
        self.gradients['weights_y'] += grad['weights_y']
        self.gradients['bias_y'] += grad['bias_y']
        if clipping:
            self.clip_grads()

    def reset_gradients(self):
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
        bias_shape = [CONTEXT_LEN, 1]
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_y': np.zeros([self.output_len, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1]),
            'output_activation': np.zeros(bias_shape),
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

    def apply_gradients(self, learning_rate=1e-3, momentum = 0.9):
        self.weights['forget'] -= learning_rate * self.gradients['weights_forget']
        self.weights['update'] -= learning_rate * self.gradients['weights_update']
        self.weights['output'] -= learning_rate * self.gradients['weights_output']
        self.weights['candidate'] -= learning_rate * self.gradients['weights_candidate']
        self.weights['y'] -= learning_rate * self.gradients['weights_y']
        self.biases['forget'] -= learning_rate * self.gradients['bias_forget']
        self.biases['update'] -= learning_rate * self.gradients['bias_update']
        self.biases['output'] -= learning_rate * self.gradients['bias_output']
        self.biases['candidate'] -= learning_rate * self.gradients['bias_candidate']
        self.biases['y'] -= learning_rate * self.gradients['bias_y']
