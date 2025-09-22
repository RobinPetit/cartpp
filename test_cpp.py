from pycart import RegressionTree
from pycart import Config, RandomForest
from pycart import print_dt
from _load_pycart_dataset import load_data

import numpy as np

DTYPE = np.float64


def area(xs, ys):
    return np.trapz(ys.reshape(-1), x=xs.reshape(-1))


(
    complete_dataset,
    dataset_train, dataset_valid,
    dataset_training, dataset_testing
) = load_data(
    DTYPE,
    ignore_categorical=False,
    frac_train=.7,
    nb_obs=1_000_000,
    kind='Wutricht',
    max_mod=10
)
LOSS = 'lorenz'


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
        if step % 50 == 0:
            ax.plot(xs, ys, label=f'After {step} splits (Gini={1-2*area(xs, ys):1.3e})')
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
        loss=LOSS, interaction_depth=51, split_type='best',
        minobs=100, dtype=DTYPE, crossing_lorenz=False,
        bootstrap=False, bootstrap_frac=.75,
        verbose=True,
        nb_covariates=0,  # nb of covariates per tree in each tree
        normalized_dloss=True
    )
    tree = RegressionTree(config)
    tree.fit(dataset_training)
    pred_train = tree.predict(dataset_training.get_X())
    # gamma, LC_training = lorenz_curve_new(
    #     dataset_training.get_y().reshape(-1),
    #     np.array(pred_train).reshape(-1)
    # )
    # gini = 1 - 2*area(gamma, LC_training)
    # print(f'Final loss: {gini:1.6f}')
    print('Gini index:', gini_index(pred_train))
    # predictions = tree.predict(dataset_testing.get_X())
    # print_dt(tree, complete_dataset)
    # show_lorenz_curves(tree)
    print(tree.get_lorenz_curves_crossings())
    print(tree.get_lorenz_curves_duplicates())
    sorted_predictions = np.asarray(
        sorted([
            leaf.pred for leaf in tree.get_all_leaves()
        ]),
        dtype=np.float64
    )
    print(sorted_predictions[1:] - sorted_predictions[:-1])

def gini_index(y_pred):
    pis, ns = np.unique(y_pred, return_counts=True)
    seq = [(0, 0)] + [(ns[i], pis[i]) for i in range(len(ns))]
    seq.sort(key=lambda x: x[1])
    x, y = zip(*seq)
    N = sum(x[0] for x in seq)
    seq = [[n/N, pred] for n, pred in seq]
    for i in range(1, len(seq)):
        seq[i][0] += seq[i-1][0]
    gamma, pi = zip(*seq)
    num = 0
    den = 0
    for i in range(1, len(seq)):
        num += (gamma[i] - gamma[i-1]) * sum((gamma[j] - gamma[j-1])*pi[j] for j in range(1, i))
        num += (gamma[i] - gamma[i-1])**2 / 2 * pi[i]
        den += (gamma[i] - gamma[i-1]) * pi[i]
    return 1 - 2*num / den

def lorenz_curve_new(y_true, y_pred):
    alpha = np.arange(y_pred.size + 1, dtype=np.float64)
    alpha /= y_pred.size
    y_pred = np.sort(y_pred)
    y_lorenz = y_pred.cumsum() / y_pred.sum()
    y_lorenz = np.insert(y_lorenz, 0, 0)
    return alpha, y_lorenz

test_dt()
# test_rfs()
