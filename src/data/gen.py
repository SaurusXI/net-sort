def generate_data(tot_samples=10000, highest=20, t=5):
    '''
    Generate data as follows:
    Alternate lines describe an input X and its prediction Y. For the pointer-net architecture,
    Y is a set of indices for the input. Each array is stored as space separated numbers
    A total of `tot_samples` number of samples is generated. `highest` is the non-inclusive upper-bound.
    `t` dentoes the total number of elements generated per sample.
    '''
    import numpy as np
    from random import seed, randint
    seed(1)
    np.random.seed(1)

    X = []
    Y = []

    for i in range(1, 1 + tot_samples):
        x = np.random.randint(1, highest, [t])
        y = np.argsort(x)
        X.append(x)
        Y.append(y)

    with open('data.txt', 'w') as f:
        for i in range(tot_samples):
            x = ' '.join([str(k) for k in X[i]])
            y = ' '.join([str(k) for k in Y[i]])
            print(x, file=f)
            print(y, file=f)
