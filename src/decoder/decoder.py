import numpy as np
from LSTM.cell import Cell
from model.utils import relu, OHE, drelu
from scipy.special import softmax


CONTEXT_LEN = 16


class Decoder:
    def __init__(self, output_len, temperature):
        self.cell = Cell()
        self.temperature = temperature
        self.output_len = output_len

        # Initialize weights and biases
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
        bias_shape = [CONTEXT_LEN, 1]
        limit = 4 * ((3 / CONTEXT_LEN) ** 0.5)
        out_limit = CONTEXT_LEN ** 0.5
        self.weights = {
            'update': np.random.default_rng().uniform(-(limit), limit, weights_shape),
            'forget': np.random.default_rng().uniform(-(limit), limit, weights_shape),
            'candidate': np.random.default_rng().uniform(-(limit) / 4, limit / 4, weights_shape),
            'output': np.random.default_rng().uniform(-(limit), limit, weights_shape),
            'W1': np.random.default_rng().uniform(-(limit) / 4, limit / 4, weights_shape),
            'W2': np.random.default_rng().uniform(-(limit) / 4, limit / 4, weights_shape),
            'v': np.random.default_rng().uniform(-out_limit, out_limit, [1, CONTEXT_LEN])
        }
        self.biases = {
            'update': np.random.random([CONTEXT_LEN, 1]),
            'forget': np.ones([CONTEXT_LEN, 1]),
            'candidate': np.random.random([CONTEXT_LEN, 1]),
            'output': np.random.random([CONTEXT_LEN, 1]),
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
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1]),
            'encoder_activations': []
        }

        self.accumulated_velocity = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1])
        }

        self.accumulated_S = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1])
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
            activation, context, cache = self.cell.forward(
                None, activation, context, self.weights, self.biases, False
            )
            ui = []
            # print(encoded_activations.shape)
            for enc_act in encoded_activations:
                uij = self.weights['v'] @ np.tanh(
                    self.weights['W1'] @ enc_act +
                    self.weights['W2'] @ activation
                )
                ui.append(uij.item())

            prediction = softmax(
                np.array(ui)
                / self.temperature
            )
            self.predictions.append(prediction)
            self.activations.append(activation)
            self.contexts.append(context)
            self.caches.append(cache)

        self.activations = np.array(self.activations)
        self.contexts = np.array(self.contexts)

        self.input_activation = encoded_activations
        self.input_context = encoded_contexts

        return self.predictions

    def backprop(self, ground_truth, encoded_activations):
        dcontext = np.zeros([CONTEXT_LEN, 1])
        dactiv_prev = np.zeros([CONTEXT_LEN, 1])

        dv = np.zeros([1, CONTEXT_LEN])
        dW1 = np.zeros([CONTEXT_LEN, 1])
        dW2 = np.zeros([CONTEXT_LEN, 1])

        self.gradients['encoder_activations'] = [
            np.zeros([CONTEXT_LEN, 1]) for i in range(self.timesteps)]

        for i in reversed(range(self.timesteps)):
            expected = OHE(ground_truth[i], self.timesteps).reshape(-1, 1)
            dui = (self.predictions[i].reshape(-1, 1) -
                   expected) / self.temperature
            ddi = np.zeros([CONTEXT_LEN, 1])
            dvj = np.zeros([1, CONTEXT_LEN])
            dW1j = np.zeros([CONTEXT_LEN, 1])
            dW2i = np.zeros([CONTEXT_LEN, 1])

            for j, duij in enumerate(dui):
                inner = np.tanh(self.weights['W1'] @ encoded_activations[j] +
                                self.weights['W2'] @ self.activations[i])
                dvj += duij.item() * inner.T
                dW1j += duij.item() * (1 - np.square(inner)) * \
                    encoded_activations[j]
                dW2i += duij.item() * (1 - np.square(inner))
                ddi += duij.item() * (1 - np.square(inner))
                self.gradients['encoder_activations'][j] += duij.item() * \
                    (1 - np.square(inner))

            dW1 += dW1j
            dW2 += dW2i * self.activations[i]
            dv += dvj
            ddi = ((self.weights['W2'] @ self.weights['v'].T)
                   * ddi) + dactiv_prev

            grad = self.cell.backprop(
                ddi,
                dcontext,
                self.caches[i],
                False
            )
            dcontext = grad['context_prev']
            dactiv_prev = grad['activ_prev']
            self.update_grads(grad)

        self.gradients['weights_W1'] = dW1 @ self.weights['v']
        self.gradients['weights_W2'] = dW2 @ self.weights['v']
        self.gradients['weights_v'] = dv
        self.gradients['encoder_activations'] = list(
            map(lambda x: x * (self.weights['W1'] @ self.weights['v'].T),
                self.gradients['encoder_activations'])
        )

        return dactiv_prev, dcontext, self.gradients['encoder_activations']

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

    def reset_gradients(self):
        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
        bias_shape = [CONTEXT_LEN, 1]
        self.gradients = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1]),
            'encoder_activations': []
        }

    def reset_accum(self):

        weights_shape = [CONTEXT_LEN, CONTEXT_LEN]
        bias_shape = [CONTEXT_LEN, 1]

        self.accumulated_velocity = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1])
        }

        self.accumulated_S = {
            'weights_forget': np.zeros(weights_shape),
            'weights_update': np.zeros(weights_shape),
            'weights_output': np.zeros(weights_shape),
            'weights_candidate': np.zeros(weights_shape),
            'weights_W1': np.zeros(weights_shape),
            'weights_W2': np.zeros(weights_shape),
            'weights_v': np.zeros([1, CONTEXT_LEN]),
            'bias_forget': np.zeros(bias_shape),
            'bias_update': np.zeros(bias_shape),
            'bias_output': np.zeros(bias_shape),
            'bias_candidate': np.zeros(bias_shape),
            'bias_y': np.zeros([self.output_len, 1])
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

    def apply_gradients(self, timestep, learning_rate=1e-3, momentum=0.9, beta=0.999, epsilon=1e-8):
        for k, v in self.weights.items():
            grad_key = 'weights_' + k
            self.accumulated_velocity[grad_key] = momentum * self.accumulated_velocity[grad_key] + \
                (1 - momentum) * self.gradients[grad_key]
            self.accumulated_S[grad_key] = beta * self.accumulated_S[grad_key] + \
                (1 - beta) * np.square(self.gradients[grad_key])

            # Correction terms
            v_corrected = self.accumulated_velocity[grad_key] / (
                1 - (momentum ** timestep))
            s_corrected = self.accumulated_S[grad_key] / \
                (1 - (beta ** timestep))

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
            v_corrected = self.accumulated_velocity[grad_key] / (
                1 - (momentum ** timestep))
            s_corrected = self.accumulated_S[grad_key] / \
                (1 - (beta ** timestep))

            self.biases[k] -= learning_rate * v_corrected / np.sqrt(
                s_corrected + epsilon
            )
