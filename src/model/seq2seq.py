from encoder.encoder import Encoder
from decoder.decoder import Decoder
from model.utils import categorical_cross_entropy, OHE
import numpy as np

CONTEXT_LEN = 16
MAX_NUM = 20


class Seq2Seq:
    def __init__(self):
        self.encoder = Encoder(MAX_NUM)
        self.decoder = Decoder(MAX_NUM, 1)

        # Forward prop stuff
        self.input = None
        self.encoded_activations = None
        self.encoded_contexts = None
        self.out = []

        # Compute loss stuff
        self.Loss = None

    def forward(self, x, debug=False):
        '''
        Forward pass through model. Forward pass through encoder generates intermediate context which is passed on to decoder for decoder forward pass.
        Furthermore encoder activations are used by decoder to make a prediction at every time-step using a modified attention mechanism for
        pointer-networks.
        '''
        self.encoded_activations, self.encoded_contexts = self.encoder.forward(
            x)
        self.timesteps = x.shape[0]
        self.out = self.decoder.forward(
            self.encoded_activations, self.encoded_contexts, self.timesteps, debug
        )
        self.input = x

    def compute_loss(self, ground_truth):
        '''
        Compute loss using categorical cross entropy
        '''
        N = np.array(ground_truth).shape[0]
        self.Loss = categorical_cross_entropy(self.out, OHE(ground_truth, N))
        self.ground_truth = ground_truth
        return self.Loss

    def backprop(self, ground_truth):
        '''
        Backpropagate through decoder and encoder to compute gradients. Since gradient activations are used for prediction by attention mechanism
        at decoder, gradients for encoder activations are also computed by decoder and backpropagated.
        '''
        dactivation_enc, dcontext_enc, enc_act_grads = self.decoder.backprop(
            ground_truth, self.encoded_activations)
        self.encoder.backprop(dactivation_enc, dcontext_enc, enc_act_grads)

    def apply_gradients(self, timestep, learning_rate=2e-2):
        '''
        Apply gradients to change weights/biases using Adam optimizer.
        '''
        self.encoder.apply_gradients(timestep, learning_rate)
        self.decoder.apply_gradients(timestep, learning_rate)

    def reset_gradients(self):
        '''
        Set encoder and decoder gradients to 0s.
        '''
        self.encoder.reset_gradients()
        self.decoder.reset_gradients()

    def reset_accum(self):
        '''
        Set encoder and decoder accumulated velocity values to 0
        '''
        self.encoder.reset_accum()
        self.decoder.reset_accum()

    def train(self, X, y, n_epochs, batch_size=1):
        '''
        Train on given data for `n_epochs` epochs.
        '''
        count = 1
        for k in range(n_epochs):
            loss = 0
            for i, x in enumerate(X):
                self.forward(x)
                loss += self.compute_loss(y[i])
                self.backprop(y[i])
                if i % batch_size == 0:
                    self.apply_gradients(count)
                    count += 1
                    self.reset_gradients()
                self.debug()
            self.apply_gradients(count)
            self.reset_gradients()
            print(f'Loss for epoch {k+1} - {loss}')
            self.forward(np.array([1, 6, 2, 4, 3]), True)
            print(
                f'Test sequence {[1, 6, 2, 4, 3]}\nPrediction {self.output()}')

    def output(self):
        return [self.input[np.argmax(i)] for i in self.out]

    def debug(self):
        with open('activations.log', 'w') as f:
            print(
                f'Encoder activations:\n{np.array(self.encoder.activations)}', file=f)
            print(
                f'Decoder activations:\n{np.array(self.decoder.activations)}', file=f)
            print(f'Predictions: {np.array(self.decoder.predictions)}', file=f)
