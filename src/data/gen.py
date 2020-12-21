def generate_data():
    import numpy as np
    from random import seed, randint
    seed(1)
    np.random.seed(1)

    X = []
    Y = []

    tot_samples = 100

    for i in range(10, tot_samples + 10):
        highest = randint(1, 10000)
        # samples = randint(1, sys.maxsize)
        x = np.random.randint(1, highest, [i])
        y = np.sort(x)
        X.append(x)
        Y.append(y)

    with open('data.txt', 'w') as f:
        for i in range(tot_samples):
            x = ' '.join([str(k) for k in X[i]])
            y = ' '.join([str(k) for k in Y[i]])
            print(x, file=f)
            print(y, file=f)
