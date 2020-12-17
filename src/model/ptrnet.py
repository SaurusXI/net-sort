from ..decoder.decoder import Decoder
from ..encoder.encoder import Encoder
from utils import OHE, cross_entropy
import numpy as np
from scipy.special import softmax


CONTEXT_LEN = 256


class PtrNet:
    def __init__(self):
        self.encoder = Encoder(1)
        self.decoder = Decoder(1)
        self.weights = {
            'encoder': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'decoder': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'reduction': np.random.random([1, CONTEXT_LEN])
        }

        # Stuff for forward prop
        self.decoder_states = None
        self.encoder_states = None
        self.x = None
        self.y = None
        self.timesteps = None

        # Stuff for compute loss
        self.labels = None
        self.losses = None
        self.Loss = None

        # Stuff for backprop
        self.gradients = {
            'u': None,
            'encoder_weight': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'decoder_weight': np.random.random([CONTEXT_LEN, CONTEXT_LEN]),
            'reduction_weight': np.random.random([1, CONTEXT_LEN]),
            'encoder_activations': [],
            'decoder_activations': []
        }

    def forward(self, x):
        encoded_activation, encoded_context = self.encoder.forward(x)
        self.timesteps = x.shape[0]
        self.decoder.forward(
            encoded_activation, encoded_context, self.timesteps
        )

        self.encoder_states = self.encoder.get_activations()
        self.decoder_states = self.decoder.get_activations()

        self.x = x
        self.y = np.array([])

        for i in range(self.timesteps):
            di = self.decoder_states[i]
            ui = np.array([])
            for j in range(self.timesteps):
                ej = self.encoder_states[j]
                z = (self.weights['encoder'] @ ej) + \
                    (self.weights['decoder'] @ di)
                uij = self.weights['reduction'] @ np.tanh(z)
                ui.append(uij[0])

            self.y.append(softmax(ui))

    def output(self):
        indices = np.argmax(self.y, axis=1)
        out = [self.x[i] for i in indices]
        return out

    def compute_loss(self, ground_truth):
        n_classes = ground_truth.shape[0]
        self.labels = np.array(
            list(map(
                lambda i: OHE(i, n_classes),
                ground_truth
            ))
        )

        self.losses = np.array([])

        for i, label in enumerate(self.labels):
            self.losses.append(
                cross_entropy(self.y[i], label)
            )

        self.Loss = np.sum(self.losses)

    def backprop(self):
        self.gradients['u'] = (self.y - self.labels) / self.timesteps

        # Compute gradients
        for i in range(self.timesteps):
            for j in range(self.timesteps):
                duij = self.gradients['u'][i][j]
                self.gradients['encoder_weight'] += duij * self.weights['reduction'].T @ ((1 - np.square(np.tanh(self.weights['encoder'] @ self.encoder_states[j] + self.weights['decoder'] @ self.decoder_states[i])).T * self.encoder_states[j].T))
                self.gradients['decoder_weight'] += duij * self.weights['reduction'].T @ ((1 - np.square(np.tanh(self.weights['encoder'] @ self.encoder_states[j] + self.weights['decoder'] @ self.decoder_states[i])).T * self.decoder_states[i].T))
                self.gradients['encoder_activations'].append(
                    duij * self.weights['reduction'].T @ ((1 - np.square(np.tanh(self.weights['encoder'] @ self.encoder_states[j] + self.weights['decoder'] @ self.decoder_states[i])).T) @ self.weights['encoder'].T)
                )
                self.gradients['decoder_activations'].append(
                    duij * self.weights['reduction'].T @ ((1 - np.square(np.tanh(self.weights['encoder'] @ self.encoder_states[j] + self.weights['decoder'] @ self.decoder_states[i])).T) @ self.weights['decoder'].T)
                )
                self.gradients['reduction_weight'] += duij * np.tanh(self.weights['encoder'] @ self.encoder_states[j] + self.weights['decoder'] @ self.decoder_states[i])

        # Normalize
        self.gradients['encoder_weight'] /= (self.timesteps ** 2)
        self.gradients['decoder_weight'] /= (self.timesteps ** 2)

        # Backprop encoder and decoder
        dactivations_enc = self.decoder.backprop(
            self.gradients['decoder_activations']
        )
        self.encoder.backprop(
            dactivations_enc, self.gradients['encoder_activations']
        )

