from decoder.decoder import Decoder
from encoder.encoder import Encoder
from model.utils import OHE, cross_entropy
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
        self.y = []

        for i in range(self.timesteps):
            di = self.decoder_states[:, :, i]
            ui = []
            for j in range(self.timesteps):
                ej = self.encoder_states[:, :, j]
                z = (self.weights['encoder'] @ ej) + \
                    (self.weights['decoder'] @ di)
                uij = self.weights['reduction'] @ np.tanh(z)
                # print(np.tanh(z))
                # print('next')
                ui.append(uij[0])

            self.y.append(
                softmax(np.array(ui))
            )

        self.y = np.array(self.y)

        return self.y

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

        self.losses = []

        for i, label in enumerate(self.labels):
            self.losses.append(
                cross_entropy(self.y[i], label)
            )

        self.losses = np.array(self.losses)
        self.Loss = np.sum(self.losses)
        return self.Loss

    def backprop(self):
        self.gradients['u'] = (self.y - self.labels)[:, :, 0] / self.timesteps

        # print(self.gradients['reduction_weight'].shape)
        # print((self.gradients['u'][0][0] * np.square(
        #         np.tanh(
        #             self.weights['encoder'] @
        #             self.encoder_states[:, :, 0] +
        #             self.weights['decoder'] @
        #             self.decoder_states[:, :, 0]
        #         )
        #     )
        # ).shape)

        # Compute gradients
        for i in range(self.timesteps):
            dej = []
            ddi = []
            for j in range(self.timesteps):
                duij = self.gradients['u'][i][j]
                z = self.weights['encoder'] @ self.encoder_states[:, :, j] + \
                    self.weights['decoder'] @ self.decoder_states[:, :, i]
                self.gradients['encoder_weight'] += (duij *
                    self.weights['reduction'].T @ ((
                            1 - np.square(np.tanh(z)).T *
                            self.encoder_states[:, :, j].T
                        )
                    )
                )
                self.gradients['decoder_weight'] += duij * self.weights['reduction'].T @ ((1 - np.square(np.tanh(z)).T * self.decoder_states[:, :, i].T))
                deij = duij * self.weights['encoder'] @ (self.weights['reduction'].T * (1 - np.square(np.tanh(z))))
                ddij = duij * self.weights['decoder'] @ (self.weights['reduction'].T * (1 - np.square(np.tanh(z))))
                dej.append(deij)
                ddi.append(ddij)

                self.gradients['reduction_weight'] += duij * np.tanh(z).T

            self.gradients['encoder_activations'].append(
                np.mean(dej, axis=0)
            )
            self.gradients['decoder_activations'].append(
                np.mean(ddi, axis=0)
            )

        # Normalize
        self.gradients['encoder_weight'] /= (self.timesteps ** 2)
        self.gradients['decoder_weight'] /= (self.timesteps ** 2)

        self.gradients['encoder_activations'] = np.array(self.gradients['encoder_activations'])
        self.gradients['decoder_activations'] = np.array(self.gradients['decoder_activations'])

        # print(f'adfadfa {self.gradients["encoder_activations"].shape}')
        # Backprop encoder and decoder
        dactivations_enc = self.decoder.backprop(
            self.gradients['decoder_activations']
        )
        self.encoder.backprop(
            dactivations_enc, self.gradients['encoder_activations']
        )

        self.gradients['encoder_activations'] = []
        self.gradients['decoder_activations'] = []

    def apply_gradients(self, learning_rate=1e3):
        self.weights['encoder'] -= learning_rate * \
            self.gradients['encoder_weight']
        self.weights['decoder'] -= learning_rate * \
            self.gradients['decoder_weight']
        self.weights['reduction'] -= learning_rate * \
            self.gradients['reduction_weight']

        self.decoder.apply_gradients(learning_rate)
        self.encoder.apply_gradients(learning_rate)

    def train(self, X, y, n_epochs):

        for k in range(n_epochs):
            for i, x in enumerate(X):
                self.forward(x)
                loss = self.compute_loss(y[i])
                self.backprop()
                self.apply_gradients()
            print(f'Loss at epoch {k} - {loss}')
