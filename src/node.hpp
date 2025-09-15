#ifndef CART_NODE_HPP
#define CART_NODE_HPP

#include <cstddef>
#include <cstdint>

#include "array.hpp"
#include "dataset.hpp"

namespace Cart {
template <typename Float>
struct Node final {
public:
    Node() = delete;

    Node(size_t id_, size_t depth_, const Dataset<Float>* dataset,
         Node<Float>* parent_=nullptr):
            id{id_},
            parent{parent_}, left_child{nullptr}, right_child{nullptr},
            depth{depth_}, nb_observations{dataset->size()},
            mean_y{mean<Float, Float>(dataset->get_y())},
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
    }

    /****** Getters ******/
    inline bool is_leaf() const {
        return left_child == nullptr and right_child == nullptr;
    }
    inline bool is_root() const {
        return parent == nullptr;
    }
    size_t id;

    Node* parent;
    Node* left_child;
    Node* right_child;

    size_t depth;
    size_t nb_observations;

    int feature_idx;
    Float loss, dloss;
    Float threshold;
    Float mean_y;

    uint64_t left_modalities{0};
    uint64_t right_modalities{0};

    const Dataset<Float>* data;
};
}

#endif  // CART_NODE_HPP
