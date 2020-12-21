from model.seq2seq import Seq2Seq
import numpy as np


sorter = Seq2Seq()
x = np.array([[1, 4, 2, 5, 6]])
y = np.array([[0, 1, 2, 3, 4]])
sorter.forward(np.array([1, 4, 2, 5, 6]))
print(sorter.out)
# sorter.compute_loss(y[0])
# sorter.backprop()
# sorter.apply_gradients()
sorter.train(x, y, 100)
print(sorter.out)
print(sorter.output())
