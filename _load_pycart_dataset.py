from pycart import Dataset

import numpy as np
from load_data import load_dataset, load_dataset_wutricht


def load_data(dtype=np.float64, verbose: bool = False, nb_obs: int = 10_000,
              ignore_categorical: bool = False,
              reduce_modalities: bool = False,
              frac_train: float = .8, kind: str = "PV", max_mod: int = 20):
    if kind == "Wutricht":
        callback = load_dataset_wutricht
    else:
        callback = load_dataset
    df_fictif, col_features, col_response, col_protected = callback(
        nb_obs=nb_obs, max_mod=max_mod, verbose=verbose
    )
    if verbose:
        print("db loaded !")

    len_or = len(df_fictif)
    df_fictif.dropna(inplace=True)
    if verbose:
        print(f"dropna ! (dropped: {len_or - len(df_fictif)}, remaining: {len(df_fictif)})")
        print(f"{col_features=}")

    # features = list(range(6, 13))
    # col_features = [col_features[j] for j in features]

    X = df_fictif[col_features].values
    y = df_fictif[col_response].values
    p = df_fictif[col_protected].values
    p = p.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)

    if ignore_categorical:
        categorical = np.asarray([
            isinstance(X[0, j], str) for j in range(X.shape[1])
        ])
        X = X[:, ~categorical]
    complete_dataset = Dataset(X, y, p, dtype=dtype)
    dataset_train, dataset_valid = complete_dataset.split(frac_train, shuffle=False)
    dataset_training, dataset_testing = dataset_train.split(frac_train, shuffle=False)
    if verbose:
        print("db splitted !")
    return (
        complete_dataset,
        dataset_train, dataset_valid,
        dataset_training, dataset_testing
    )
