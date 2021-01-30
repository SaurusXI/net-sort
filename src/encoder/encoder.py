import numpy as np
from LSTM.cell import Cell
from model.utils import OHE


CONTEXT_LEN = 16


class Encoder:
    def __init__(self, input_len):
        self.cell = Cell()
        self.input_len = input_len
        # Initialize weights and biases
        limit = 4 * ((6 / (2*CONTEXT_LEN + input_len)) ** 0.5)
        self.weights = {
            'update': np.random.default_rng().uniform(-limit, limit, [CONTEXT_LEN, CONTEXT_LEN + input_len]),
            'forget': np.random.default_rng().uniform(-limit, limit, [CONTEXT_LEN, CONTEXT_LEN + input_len]),
            'candidate': np.random.default_rng().uniform(-limit / 4, limit / 4, 
                [CONTEXT_LEN, CONTEXT_LEN + input_len]
            ),
            'output': np.random.default_rng().uniform(-limit, limit, [CONTEXT_LEN, CONTEXT_LEN + input_len])
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

        self.accumulated_velocity = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
        }

        self.accumulated_S = {
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
            activation, context, cache = self.cell.forward(
                x[t], activation, context, self.weights, self.biases
            )
            self.activations.append(activation)
            self.contexts.append(context)
            self.caches.append(cache)

        self.input = x
        self.activations = np.array(self.activations)
        self.contexts = np.array(self.contexts)

        return self.activations, self.contexts

    def backprop(self, dactiv_prev, dcontext, dactivations):
        timesteps = np.array(self.input).shape[0]

        for t in reversed(range(timesteps)):
            dactiv_prev += dactivations[t]
            grad = self.cell.backprop(
                dactiv_prev,
                dcontext,
                self.caches[t]
            )
            dactiv_prev = grad['activ_prev']
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

    def reset_accum(self):
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN + self.input_len]
        bias_shape = [CONTEXT_LEN, 1]

        self.accumulated_velocity = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
        }

        self.accumulated_S = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
        }

    def apply_gradients(self, timestep, learning_rate=1e-3, momentum=0.9, beta = 0.999, epsilon = 1e-8):
        for k, v in self.weights.items():
            grad_key = 'weights_' + k
            self.accumulated_velocity[grad_key] = momentum * self.accumulated_velocity[grad_key] + \
                (1 - momentum) * self.gradients[grad_key]
            self.accumulated_S[grad_key] = beta * self.accumulated_S[grad_key] + \
                (1 - beta) * np.square(self.gradients[grad_key])

            # Correction terms
            v_corrected = self.accumulated_velocity[grad_key] / (1 - (momentum ** timestep))
            s_corrected = self.accumulated_S[grad_key] / (1 - (beta ** timestep))

            self.weights[k] -= learning_rate * v_corrected / np.sqrt(
                s_corrected + epsilon
            )

        for k, v in self.biases.items():
            grad_key = 'bias_' + k
            self.accumulated_velocity[grad_key] = momentum * self.accumulated_velocity[grad_key] + \
                (1 - momentum) * self.gradients[grad_key]
            self.accumulated_S[grad_key] = beta * self.accumulated_S[grad_key] + \
                (1 - beta) * np.square(self.gradients[grad_key])

            # Correction terms
            v_corrected = self.accumulated_velocity[grad_key] / (1 - (momentum ** timestep))
            s_corrected = self.accumulated_S[grad_key] / (1 - (beta ** timestep))

            self.biases[k] -= learning_rate * v_corrected / np.sqrt(
                s_corrected + epsilon
            )