# TODO

## Write unit tests with `Catch2`

## Handle random forests

Use C++ threading to dispatch training and prediction of independent trees.

## Fairness criteria

See e.g. [DPTree and DPForest: tree-based methods fulfilling demographic parity](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/dptree-and-dpforest-treebased-methods-fulfilling-demographic-parity/1D811F0D2ED70E8753FDB7839843E169).

## Separate `cartpp` from the bindings

Probably to be done after publication of the paper.

## Oblique trees

Instead of splitting on a single covariate at each node, use a linear combination for the split.
Long term plan.

### Implementation idea

If `Splitter` classes define a `typedef <...> NodeType` where `NodeType` inherits from `BaseNode` (currently `Node`)
contains the appropriate informations for the type of split (typically best first, depth first but also e.g. the number of covariates to consider in each split, 1 by default),
then it would be possible to keep the `Tree` classes pretty much intact but allow different types of constructions.
