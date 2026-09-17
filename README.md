# Cart++

The library is header-only (and heavily relies on C++20 templates and concepts),
so just include the appropriate files when compiling.

## Features

In `cartpp`, one will find an implementation of CART for regression and classification trees.
The existing loss functions so far are:
- MSE
- Poisson deviance
- Negative binomial deviance

In addition to _classical_ CART, `cartpp` proposes an implementation of _Gini-trees_ (TODO: add link to paper).
That is, in addition to a construction of trees where the loss at each leaf is independent of the loss at every other leaf;
`cartpp` proposes a loss at each leaf that depends on the whole tree (the distribution of predictions).
That loss is the _Gini index_ of the tree.

## Bindings

### Python binding

The module `pycart` is a Python binding using `Cart++`.
The binding is written using `Cython`.

### R binding

Will come at some point (maybe?)


## TODO

See [`TODO.md`](https://github.com/RobinPetit/cartpp/blob/main/TODO.md)
