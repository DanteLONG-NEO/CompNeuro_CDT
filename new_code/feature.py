import numpy as np

def make_lagged_matrix(X, lags):
    """
    X: [T, D]
    lags: list/array of ints
    return: [T, D * len(lags)]
    """
    X = np.asarray(X, dtype=float)
    T, D = X.shape
    mats = []

    for lag in lags:
        Xs = np.zeros((T, D), dtype=float)

        if lag < 0:
            Xs[:lag, :] = X[-lag:, :]
        elif lag > 0:
            Xs[lag:, :] = X[:-lag, :]
        else:
            Xs[:, :] = X

        mats.append(Xs)

    return np.concatenate(mats, axis=1)