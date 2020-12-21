from encoder.encoder import Encoder
from decoder.decoder import Decoder
from model.utils import relu, cross_entropy
import numpy as np


CONTEXT_LEN = 256


class Seq2Seq:
    def __init__(self):
        self.encoder = Encoder(1)
        self.decoder = Decoder(1)

        # Forward prop stuff
        self.weights = {'output': np.random.random([1, CONTEXT_LEN])}
        self.bias = {'output': np.random.random_sample(size=None)}
        self.input = None
        self.out = []

        # Compute loss stuff
        self.Loss = None

        # Compute backprop stuff
        self.gradients = {
            'output': None,
            'weights_output': np.zeros([1, CONTEXT_LEN]),
            'bias_output': 0,
            'decoder_activation': []
        }

    def forward(self, x):
        encoded_activation, encoded_context = self.encoder.forward(x)
        self.timesteps = x.shape[0]
        self.out = []
        self.decoder_activations = self.decoder.forward(
            encoded_activation, encoded_context, self.timesteps
        )

        for t in range(self.timesteps):
            activation = self.decoder_activations[:, :, t]
            self.out.append(
                relu(
                    (self.weights['output'] @ activation) + self.bias['output']
                )[0][0]
            )

        self.input = x
        self.out = np.array(self.out)

    def compute_loss(self, ground_truth):
        self.Loss = cross_entropy(self.out, ground_truth)
        self.ground_truth = ground_truth
        return self.Loss

    def backprop(self):
        for i, output in enumerate(self.out):
            zi = ((self.weights['output'] @ self.decoder_activations[:, :, i])
                  + self.bias['output'])[0]
            drelu = 1 if zi > 0 else 0
            doi = -(self.ground_truth[i]/output)

            self.gradients['weights_output'] += doi * drelu * self.decoder_activations[:, :, i].T
            self.gradients['bias_output'] += doi * drelu
            self.gradients['decoder_activation'].append(
                doi * drelu * self.weights['output'].T
            )

        self.gradients['decoder_activation'] = np.array(
            self.gradients['decoder_activation']
        )

        dactivation_enc, dcontext_enc = \
            self.decoder.backprop(self.gradients['decoder_activation'])
        self.encoder.backprop(dactivation_enc, dcontext_enc)

        self.gradients['decoder_activation'] = []

    def apply_gradients(self, learning_rate=1e-1):
        self.weights['output'] -= learning_rate * self.gradients['weights_output']
        self.bias['output'] -= learning_rate * self.gradients['bias_output']
        self.encoder.apply_gradients(learning_rate)
        self.decoder.apply_gradients(learning_rate)

    def reset_gradients(self):
        self.gradients = {
            'output': None,
            'weights_output': np.zeros([1, CONTEXT_LEN]),
            'bias_output': 0,
            'decoder_activation': []
        }
        self.encoder.reset_gradients()
        self.decoder.reset_gradients()

    def train(self, X, y, n_epochs):

        for k in range(n_epochs):
            loss = 0
            for i, x in enumerate(X):
                self.forward(x)
                loss += self.compute_loss(y[i])
                print(f'Loss for sample {i} - {loss}')
                self.backprop()
                self.apply_gradients()
            self.reset_gradients()
            print(f'Loss at epoch {k} - {loss}')

    def output(self):
        return [round(i) for i in self.out]
