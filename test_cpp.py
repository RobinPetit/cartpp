from pycart import RegressionTree, Config
from _load_pycart_dataset import load_data

import numpy as np

DTYPE = np.float64

dataset = load_data(DTYPE)

config = Config(
    loss='poisson', interaction_depth=5, split_type='depth',
    minobs=10, dtype=DTYPE
)
tree = RegressionTree(config)
tree.fit(dataset)
