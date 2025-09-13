from pycart import RegressionTree, Config, RandomForest
from _load_pycart_dataset import load_data

import matplotlib.pyplot as plt

import numpy as np

DTYPE = np.float64


def area(xs, ys):
    return np.trapz(ys, xs)


dataset, test = load_data(
    DTYPE, ignore_categorical=False,
    reduce_modalities=True,
    nb_obs=10_000_000,
    frac_train=.99,
)


def show_lorenz_curves(model):
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


Model = RegressionTree
# Model = lambda cfg: RandomForest(cfg, 100, 10)

LOSS = 'lorenz'

config = Config(
    loss=LOSS, interaction_depth=101, split_type='best',
    minobs=3000, dtype=DTYPE, crossing_lorenz=True,
    # bootstrap=True, bootstrap_frac=.5,
    verbose=Model is RegressionTree,
    nb_covariates=8  # nb of covariates per tree in each tree
)
model = Model(config)
model.fit(dataset)
print(dataset.get_X().shape)
print(test.get_X().shape)
predictions = model.predict(test.get_X())
print(model.get_feature_importance(dataset.get_X().shape[1]))
if Model is RegressionTree:
    show_lorenz_curves(model)
