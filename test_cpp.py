from pycart import RegressionTree, Config
from _load_pycart_dataset import load_data

import matplotlib.pyplot as plt

import numpy as np

DTYPE = np.float64


def area(xs, ys):
    return np.trapz(ys, xs)


dataset, test = load_data(
    DTYPE, ignore_categorical=False,
    reduce_modalities=True, nb_obs=20_000
)

config = Config(
    loss='lorenz', interaction_depth=21, split_type='best',
    minobs=10, dtype=DTYPE,
)
tree = RegressionTree(config)
tree.fit(dataset)
predictions = tree.predict(test.get_X())
lcs = tree.get_lorenz_curves()
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
plt.savefig('lorenz_curves.png', bbox_inches='tight')
