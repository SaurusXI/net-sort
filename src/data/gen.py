def generate_data():
    import numpy as np
    from random import seed, randint
    seed(1)
    np.random.seed(1)

    X = []
    Y = []

    tot_samples = 10000

    for i in range(1, 1 + tot_samples):
        highest = 10
        # samples = randint(1, sys.maxsize)
        t = 1 if i < 1000 else i // 1000
        x = np.random.randint(1, highest, [t])
        y = np.sort(x)
        X.append(x)
        Y.append(y)

    with open('data.txt', 'w') as f:
        for i in range(tot_samples):
            x = ' '.join([str(k) for k in X[i]])
            y = ' '.join([str(k) for k in Y[i]])
            print(x, file=f)
            print(y, file=f)
