from encoder.encoder import Encoder
from decoder.decoder import Decoder
from model.utils import cross_entropy


CONTEXT_LEN = 256


class Seq2Seq:
    def __init__(self):
        self.encoder = Encoder(1)
        self.decoder = Decoder(1)

        # Forward prop stuff
        self.input = None
        self.out = []

        # Compute loss stuff
        self.Loss = None

    def forward(self, x):
        encoded_activations, encoded_contexts = self.encoder.forward(x)
        self.timesteps = x.shape[0]
        self.out = self.decoder.forward(
            encoded_activations, encoded_contexts, self.timesteps
        )
        self.input = x

    def compute_loss(self, ground_truth):
        self.Loss = cross_entropy(self.out, ground_truth)
        self.ground_truth = ground_truth
        return self.Loss

    def backprop(self, ground_truth):
        dactivations_enc, dcontext_enc = self.decoder.backprop(ground_truth)
        self.encoder.backprop(dactivations_enc, dcontext_enc)

    def apply_gradients(self, learning_rate=1e-7):
        self.encoder.apply_gradients(learning_rate)
        self.decoder.apply_gradients(learning_rate)

    def reset_gradients(self):
        self.encoder.reset_gradients()
        self.decoder.reset_gradients()

    def train(self, X, y, n_epochs):

        for k in range(n_epochs):
            loss = 0
            for i, x in enumerate(X):
                self.forward(x)
                loss += self.compute_loss(y[i])
                self.backprop(y[i])
                self.apply_gradients()
                self.reset_gradients()
                print(f'Loss for sample {i} - {loss}')
            print(f'Loss at epoch {k} - {loss}')

    def output(self):
        return [round(i) for i in self.out]
