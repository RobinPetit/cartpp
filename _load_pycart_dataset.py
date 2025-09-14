from pycart import Dataset

import numpy as np
from load_data import *


def load_data(dtype=np.float64, verbose: bool = False, nb_obs: int = 10_000,
              ignore_categorical: bool = False,
              reduce_modalities: bool = False,
              frac_train: float = .7):
    df_fictif, col_features, col_response, col_protected = load_dataset(
        nb_obs=nb_obs, verbose=verbose, max_mod=10
    )
    df_fictif.dropna(inplace=True)

    # features = list(range(6, 13))
    # col_features = [col_features[j] for j in features]

    df_train = df_fictif.iloc[:int(len(df_fictif)*frac_train), :]
    df_test = df_fictif.iloc[int(len(df_fictif)*frac_train):, :]

    # Splitting adequately the sets
    X_train = df_train[col_features].values
    y_train = df_train[col_response].values
    p_train = df_train[col_protected].values
    p_train = p_train.astype(np.float64).reshape(-1)
    y_train = y_train.astype(np.float64).reshape(-1)

    X_test = df_test[col_features].values
    y_test = df_test[col_response].values
    p_test = df_test[col_protected].values
    p_test = p_test.astype(np.float64).reshape(-1)
    y_test = y_test.astype(np.float64).reshape(-1)

    X_train = X_train.astype(object)
    X_test = X_test.astype(object)

    if ignore_categorical:
        categorical = np.asarray([
            isinstance(X_train[0, j], str) for j in range(X_train.shape[1])
        ])
        X_train = X_train[:, ~categorical]
        X_test = X_test[:, ~categorical]
    else:
        if reduce_modalities:
            # Just to reduce the number of modalities for exact split  #
            values, counts = np.unique(X_train[:, 2], return_counts=True)
            values, counts = zip(*sorted(
                list(zip(values, counts)),
                key=lambda e: e[1])
            )
            cdf = np.cumsum(counts) * 100 / X_train.shape[0]
            idx = np.where(cdf >= 8.)[0][0]
            indices = np.zeros(X_train.shape[0], dtype=bool)
            for i in range(idx, len(values)):
                indices[X_train[:, 2] == values[i]] = True
            X_train = X_train[indices, :]
            y_train = y_train[indices]
            p_train = p_train[indices]

    ret_train = Dataset(X_train, y_train, p_train, dtype=dtype)
    ret_test = Dataset(X_test, y_test, p_test, dtype=dtype)
    return (ret_train, ret_test)
