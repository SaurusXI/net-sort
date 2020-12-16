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
            'encoder': np.random.random([1, CONTEXT_LEN]),
            'decoder': np.random.random([1, CONTEXT_LEN])
        }
        self.bias = np.random.random([1, 1])

    def forward(self, x):
        encoded_activation, encoded_context = self.encoder.forward(x)
        timesteps = x.shape[0]
        self.decoder.forward(encoded_activation, encoded_context, timesteps)

        encoder_states = self.encoder.get_activations()
        decoder_states = self.decoder.get_activations()

        self.y = np.array([])

        for i in range(timesteps):
            di = decoder_states[i]
            ui = np.array([])
            for j in range(timesteps):
                ej = encoder_states[j]
                z = (self.weights['encoder'] @ ej) + (self.weights['decoder'] @ di) + self.bias
                uij = np.tanh(z)
                ui.append(uij[0])

            self.y.append(softmax(ui))

