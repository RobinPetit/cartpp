from pycart import RegressionTree, Dataset, Config

import numpy as np

from load_data import load_dataset

VERBOSE = False
nb_observation = 10_000

df_fictif, col_features, col_response, col_protected = load_dataset(
    nb_obs=nb_observation, verbose=VERBOSE
)
df_fictif.dropna(inplace=True)

# features = list(range(6, 13))
# col_features = [col_features[j] for j in features]

frac_train = 0.7

df_train = df_fictif.iloc[:int(len(df_fictif)*frac_train), :]
df_test = df_fictif.iloc[int(len(df_fictif)*frac_train):, :]

df_training = df_train.iloc[:int(len(df_train)*frac_train), :]
df_testing = df_train.iloc[int(len(df_train)*frac_train):, :]

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

X_training = df_training[col_features].values
y_training = df_training[col_response].values
p_training = df_training[col_protected].values
p_training = p_training.astype(np.float64).reshape(-1)
y_training = y_training.astype(np.float64).reshape(-1)

X_testing = df_testing[col_features].values
y_testing = df_testing[col_response].values
p_testing = df_testing[col_protected].values
p_testing = p_testing.astype(np.float64).reshape(-1)
y_testing = y_testing.astype(np.float64).reshape(-1)

dtypes = df_fictif[col_features].dtypes.values
dic_cov = {col: str(df_fictif[col].dtype) for col in col_features}

X_train = X_train.astype(object)
X_test = X_test.astype(object)

# Just to reduce the number of modalities for exact split  #
values, counts = np.unique(X_train[:, 2], return_counts=True)
values, counts = zip(*sorted(list(zip(values, counts)), key=lambda e: e[1]))
cdf = np.cumsum(counts) * 100 / X_train.shape[0]
idx = np.where(cdf >= 5.)[0][0]
indices = np.zeros(X_train.shape[0], dtype=bool)
for i in range(idx, len(values)):
    indices[X_train[:, 2] == values[i]] = True
X_train = X_train[indices, :]
y_train = y_train[indices]
p_train = p_train[indices]

DTYPE = np.float32

dataset = Dataset(X_train, y_train, p_train, dtype=DTYPE)
config = Config(loss='poisson', interaction_depth=5, split_type='depth', minobs=10, dtype=DTYPE)
tree = RegressionTree(config)
if False:
    tree.fit(dataset)
