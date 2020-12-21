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
        self.weights['output'] = np.random.random([1, CONTEXT_LEN])
        self.bias['output'] = np.random.random_sample(size=None)
        self.out = []

        # Compute loss stuff

    def forward(self, x):
        encoded_activation, encoded_context = self.encoder.forward(x)
        self.timesteps = x.shape[0]
        decoder_activations = self.decoder.forward(
            encoded_activation, encoded_context, self.timesteps
        )

        for activation in decoder_activations:
            self.out.append(
                relu(
                    (self.weights['output'] @ activation) + self.bias['output']
                )
            )

    def compute_loss(self, ground_truth):
        self.Loss = cross_entropy(self.out, ground_truth)
        return self.Loss
