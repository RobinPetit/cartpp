#ifndef CART_NODE_HPP
#define CART_NODE_HPP

#include <cstddef>
#include <cstdint>

#include "array.hpp"
#include "dataset.hpp"

namespace Cart {
/**
 * @brief Node of regression-/decision-trees.
 *
 * Should not be used explicitly, go through Regression::BaseRegressionTree.
 */
template <typename Float>
struct Node final {
public:
    Node() = delete;

    Node(size_t id_, size_t depth_, const Dataset<Float>* dataset,
         Node<Float>* parent_=nullptr):
            id{id_},
            parent{parent_}, left_child{nullptr}, right_child{nullptr},
            depth{depth_}, nb_observations{dataset->size()},
            sum_of_weights{
                dataset->is_weighted()
                ? dataset->weighted_size()
                : static_cast<Float>(dataset->size())
            },
            pred{
                dataset->is_weighted()
                ? weighted_mean<Float>(dataset->get_y(), dataset->get_w())
                : mean<Float, Float>(dataset->get_y())
            },
            data{dataset} {
    }

    ~Node() {
        if(left_child != nullptr) {
            delete left_child;
            left_child = nullptr;
        }
        if(right_child != nullptr) {
            delete right_child;
            right_child = nullptr;
        }
        if(parent != nullptr) {
            if(parent->left_child == this)
                parent->left_child = nullptr;
            else
                parent->right_child = nullptr;
        }
        if(data != nullptr) {
            if(parent != nullptr)
                delete data;  // root is not owner of its dataset
            data = nullptr;
        }
        parent = nullptr;
    }

    /****** Getters ******/
    inline bool is_leaf() const {
        return left_child == nullptr and right_child == nullptr;
    }
    inline bool is_root() const {
        return parent == nullptr;
    }

    /// Unique identifier of the node
    size_t id;

    /// Pointer to the parent (nullptr for the root).
    Node* parent;
    /// Pointer to the left child.
    Node* left_child;
    /// Pointer to the right child.
    Node* right_child;

    /// Depth of the node (distance to root).
    size_t depth;
    /// Number of observations within that node.
    size_t nb_observations;
    /// Sum of the weights of the observations within that node.
    Float sum_of_weights;

    /// Index of the covariate used to split this internal node.
    int feature_idx;
    /// Loss of the observations in this node (before split).
    Float loss;
    /// Dloss of the associated split.
    Float dloss;
    /// Threshold of the non-categorical covariate used for the split.
    Float threshold;
    /// Weighted average of the observations within that node.
    Float pred;

    /// Mask of modalities that go to the left child (if the split is done on a categorical covariate).
    uint64_t left_modalities{0};
    /// Mask of modalities that go to the right child (if the split is done on a categorical covariate).
    uint64_t right_modalities{0};

    /// Dataset containing all observations in that node. Cleared and reset to nullptr after construction of the tree.
    const Dataset<Float>* data;

    /// Array of indices of available features to perform a split
    Array<size_t> features;
    /// Whether of not Node::features has been computed.
    bool computed_features{false};
};
}

#endif  // CART_NODE_HPP
