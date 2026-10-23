#ifndef CART_CONFIG_HPP
#define CART_CONFIG_HPP

#include <cstddef>
#include <limits>

namespace Cart {
enum class NodeSelector {
    BEST_FIRST,
    DEPTH_FIRST
};

union AdditionalParams {
    struct { double alpha; } _nb;
};

/**
 * @brief Configuration for the creation of decision trees.
 */
struct TreeConfig {
    /// Use bootstrapping for the training set
    bool bootstrap = false;
    /// Proportion of the training set used for bootstrapping
    double bootstrap_frac = 1.;
    /// Allow replacement for bootstrapping
    bool bootstrap_replacement = true;
    /// TODO: document and clarify the role.
    bool exact_splits = true;
    /// How to construct the decision tree
    NodeSelector split_type = NodeSelector::BEST_FIRST;
    /// Limit on the depth of the created tree
    size_t max_depth = std::numeric_limits<size_t>::max();
    /// Limit on the number of internal nodes in the tree
    size_t interaction_depth = std::numeric_limits<size_t>::max();
    /// Minimum size of a subdataset to be splitted
    size_t minobs = 1;
    /// Display additional info on stdout
    bool verbose = false;
    /// Number of covariates subsampled at each node (0 for no limit)
    size_t nb_covariates = 0;
    /// Normalise the Δloss by the size of the dataset
    bool normalized_dloss = true;
    /// UNUSED
    double prop_validation = 0.;

    /// Additional loss-specific parameters
    AdditionalParams _params;
};
}

#endif  // CART_CONFIG_HPP
