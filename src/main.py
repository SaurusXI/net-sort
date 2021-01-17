from model.seq2seq import Seq2Seq
import numpy as np
from data.gen import generate_data

X = []
Y = []

k = 0
generate_data()
with open('data.txt', 'r') as f:
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

N_EPOCHS = 6
SEQ_TO_SORT = [1, 4, 2, 6, 5]


if __name__ == '__main__':
    sorter = Seq2Seq()
    sorter.train(X, Y, N_EPOCHS, 512)
    sorter.forward(np.array(SEQ_TO_SORT))
    print(f'Sorted sequence: {sorter.output()}')
