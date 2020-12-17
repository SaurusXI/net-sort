from model.ptrnet import PtrNet
import numpy as np


sorter = PtrNet()
x = np.array([1, 4, 2, 5, 6])
print(sorter.forward(x))
