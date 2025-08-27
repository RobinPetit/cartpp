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
    bool exact_splits = true;
    NodeSelector split_type = NodeSelector::BEST_FIRST;
    size_t max_depth = std::numeric_limits<size_t>::max();
    size_t interaction_depth = std::numeric_limits<size_t>::max();
    size_t minobs = 1;
};
}

#endif  // CART_CONFIG_HPP
