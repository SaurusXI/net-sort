from model.ptrnet import PtrNet
import numpy as np


sorter = PtrNet()
x = np.array([[1, 4, 2, 5, 6]])
y = np.array([[0, 1, 2, 3, 4]])
sorter.train(x, y, 100)
sorter.forward(np.array([1, 4, 2, 5, 6]))
print(sorter.output())
