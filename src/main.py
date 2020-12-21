from model.seq2seq import Seq2Seq
import numpy as np
import os

X = []
Y = []

k = 0
with open(os.path.join('..', 'data', 'data.txt'), 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if k % 2 == 0:
            x = np.asarray(list(map(int, line.split())), dtype=np.int32)
            X.append(x)
        else:
            y = np.asarray(list(map(int, line.split())), dtype=np.int32)
            Y.append(y)
        k += 1

sorter = Seq2Seq()
# x = np.asarray(X, dtype=np.int32)
# y = np.asarray(Y, dtype=np.int32)
# print(x)
# sorter.forward(np.array([1, 4, 2, 5, 6]))
# print(sorter.out)
# # sorter.compute_loss(y[0])
# # sorter.backprop()
# # sorter.apply_gradients()
sorter.train(X, Y, 4)
# print(sorter.out)
sorter.forward(X[3])
print(X[3])
print(sorter.output())
