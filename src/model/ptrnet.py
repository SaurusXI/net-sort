from ..decoder.decoder import Decoder
from ..encoder.encoder import Encoder
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

    def forward(self, x):
        encoded_activation, encoded_context = self.encoder.forward(x)
        timesteps = x.shape[0]
        self.decoder.forward(encoded_activation, encoded_context, timesteps)

        self.encoder_states = self.encoder.get_activations()
        self.decoder_states = self.decoder.get_activations()

        self.x = x
        self.y = np.array([])

        for i in range(timesteps):
            di = self.decoder_states[i]
            ui = np.array([])
            for j in range(timesteps):
                ej = self.encoder_states[j]
                z = (self.weights['encoder'] @ ej) + \
                    (self.weights['decoder'] @ di) + self.bias
                uij = self.weights['reduction'] @ np.tanh(z)
                ui.append(uij[0])

            self.y.append(softmax(ui))

    def output(self):
        indices = np.argmax(self.y, axis=1)
        out = [self.x[i] for i in indices]
        return out
