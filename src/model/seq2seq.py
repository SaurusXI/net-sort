from encoder.encoder import Encoder
from decoder.decoder import Decoder
from model.utils import categorical_cross_entropy, OHE
import numpy as np

CONTEXT_LEN = 64
MAX_NUM = 10


class Seq2Seq:
    def __init__(self):
        self.encoder = Encoder(MAX_NUM)
        self.decoder = Decoder(MAX_NUM, 0.5)

        # Forward prop stuff
        self.input = None
        self.out = []

        # Compute loss stuff
        self.Loss = None

    def forward(self, x, debug=False):
        encoded_activations, encoded_contexts = self.encoder.forward(x)
        self.timesteps = x.shape[0]
        self.out = self.decoder.forward(
            encoded_activations, encoded_contexts, self.timesteps, debug
        )
        self.input = x

    def compute_loss(self, ground_truth):
        self.Loss = categorical_cross_entropy(self.out, OHE(ground_truth, MAX_NUM))
        self.ground_truth = ground_truth
        return self.Loss

    def backprop(self, ground_truth):
        dactivation_enc, dcontext_enc = self.decoder.backprop(ground_truth)
        self.encoder.backprop(dactivation_enc, dcontext_enc)

    def apply_gradients(self, learning_rate=1e-4):
        self.encoder.apply_gradients(learning_rate)
        self.decoder.apply_gradients(learning_rate)

    def reset_gradients(self):
        self.encoder.reset_gradients()
        self.decoder.reset_gradients()

    def train(self, X, y, n_epochs, batch_size=1):
        # self.forward(np.array([1, 6, 2, 4, 3]))
        for k in range(n_epochs):
            loss = 0
            # print(f'Training epoch {k + 1} [', end='')
            for i, x in enumerate(X):
                self.forward(x)
                loss += self.compute_loss(y[i])
                self.backprop(y[i])
                if i % batch_size == 0:
                    self.apply_gradients()
                    self.reset_gradients()
                # if i % 100 == 0:
                    # print('-', end='')
                # print(f'Loss at sample {i+1} - {loss}')
            self.apply_gradients()
            self.reset_gradients()
            # print(']')
            print(f'Loss for epoch {k+1} - {loss}')
            self.forward(np.array([1, 6, 2, 4, 3]), True)
            print(f'Test sequence {[1, 6, 2, 4, 3]}\nPrediction {self.output()}')

    def output(self):
        return [round(np.argmax(i)) for i in self.out]
