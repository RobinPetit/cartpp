from pycart import RegressionTree, Config, RandomForest
from tmp import load_data

import matplotlib.pyplot as plt

import numpy as np

DTYPE = np.float64


def area(xs, ys):
    return np.trapz(ys, xs)


PATH_SAVE = "RESULTS/"

KIND = "Wutricht" # "PV" #
FRAC_TRAIN = .8

MAX_MOD = 15

# dataset_training, dataset_testing, dataset_train, dataset_valid, dataset, df_tot, col_features, col_response, col_protected
(
    complete_dataset,
    dataset_train, dataset_valid,
    dataset_training, dataset_testing
) = load_data(
    DTYPE, ignore_categorical=False,
    reduce_modalities=False, nb_obs=1_000_000, kind=KIND, max_mod=MAX_MOD,
    frac_train=FRAC_TRAIN
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
#Model = lambda cfg: RandomForest(cfg, nb_trees=10, n_jobs = 10)

LOSS = 'poisson' #'lorenz' #

config = Config(
    loss=LOSS, interaction_depth=7, split_type='best',
    minobs=10, dtype=DTYPE, crossing_lorenz=False,
    bootstrap=False, bootstrap_frac=0, verbose=True
)

model = Model(config)
print("model defined, ready to fit !")
# print(col_features)
model.fit(dataset_train)

predictions_tree = model.predict(dataset_valid.get_X())
print(predictions_tree)

#
model_RF = RandomForest(config, nb_trees=10, n_jobs=10)
model_RF.fit(dataset_train)
predictions_RF = model_RF.predict_incremental(dataset_valid.get_X())

print(np.unique(predictions_tree, return_counts=True))
print(np.unique(predictions_RF[:, -1], return_counts=True))



quit()

#print(dataset_train.get_X().shape)
#print(dataset_valid.get_X().shape)


print(model.get_feature_importance(dataset.get_X().shape[1]))
if Model is RegressionTree:
    show_lorenz_curves(model)

