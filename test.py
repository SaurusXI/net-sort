import numpy as np

l = []

for i in range(3):
    t = []
    for j in reversed(range(3)):
        a = np.array([[i, j], [j, i]])
        t.append(a)

    l.append(np.mean([t], axis=0))

print(np.array(l).shape)
