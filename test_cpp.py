from pycart import RegressionTree
from pycart import Config, RandomForest
from _load_pycart_dataset import load_data

import numpy as np

DTYPE = np.float64


def area(xs, ys):
    return np.trapz(ys, xs)


(
    complete_dataset,
    dataset_train, dataset_valid,
    dataset_training, dataset_testing
) = load_data(
    DTYPE,
    ignore_categorical=False,
    frac_train=.7,
    nb_obs=1_000_000,
    kind='PV',
    max_mod=10
)
LOSS = 'poisson'


def show_lorenz_curves(model):
    import matplotlib.pyplot as plt

    lcs = model.get_lorenz_curves()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    step = 0
    idx = 0
    length = 4
    while idx < lcs.size:
        xs, ys = lcs[idx:idx+length].reshape(-1, 2).T
        if step % 20 == 0:
            ax.plot(xs, ys, label=f'After {step} splits (={.5-area(xs, ys):1.3e})')
        step += 1
        idx += length
        length += 2
    ax.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(f'lorenz_curves_{LOSS}.png', bbox_inches='tight')


def test_rfs():
    from time import time
    for nb_cov in range(1, 16):
        config = Config(
            loss=LOSS, interaction_depth=10, split_type='best',
            minobs=100, dtype=DTYPE, crossing_lorenz=False,
            bootstrap=True, bootstrap_frac=.5,
            verbose=False,
            nb_covariates=nb_cov,  # nb of covariates per tree in each tree
        )
        NB_TREES = 1000
        rf = RandomForest(config, NB_TREES, -1)
        beg = time()
        rf.fit(dataset_testing)
        end = time()
        print('Training', NB_TREES, 'trees on', nb_cov,
              'covariate(s) took', f'{end-beg:3.2f}s')


def test_dt():
    config = Config(
        loss=LOSS, interaction_depth=21, split_type='best',
        minobs=100, dtype=DTYPE, crossing_lorenz=True,
        bootstrap=True, bootstrap_frac=.5,
        verbose=True,
        nb_covariates=0,  # nb of covariates per tree in each tree
        normalized_dloss=True
    )
    tree = RegressionTree(config)
    tree.fit(dataset_training)
    predictions = tree.predict(dataset_testing.get_X())


# test_dt()
test_rfs()
