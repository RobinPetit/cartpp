#ifndef CART_CONFIG_HPP
#define CART_CONFIG_HPP

#include <cstddef>
#include <limits>

namespace Cart {
enum class NodeSelector {
    BEST_FIRST,
    DEPTH_FIRST
};

struct TreeConfig {
    bool bootstrap = false;
    double bootstrap_frac = 1.;
    bool bootstrap_replacement = true;
    bool exact_splits = true;
    NodeSelector split_type = NodeSelector::BEST_FIRST;
    size_t max_depth = std::numeric_limits<size_t>::max();
    size_t interaction_depth = std::numeric_limits<size_t>::max();
    size_t minobs = 1;
    bool verbose = false;
    size_t nb_covariates = 0;
    bool normalized_dloss = true;

    /* Loss-specific parameters */
    union {
        // Negative Binomial
        struct {
            double alpha;
        } _nb;
    } _params;
};
}

#endif  // CART_CONFIG_HPP
