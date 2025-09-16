#ifndef CART_TREE_HPP
#define CART_TREE_HPP

#include <chrono>
#include <cmath>
#include <concepts>
#include <iomanip>
#include <stdexcept>

#include "array.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "node.hpp"
#include "splitter.hpp"

namespace Cart {
namespace Regression {
template <
    std::floating_point Float,
    typename LossType
>
class BaseRegressionTree {
public:
    class Iterator {
    public:
        Iterator(Node<Float>* root):
                stack() {
            if(root != nullptr)
                stack.push(root);
        }
        inline Iterator& operator++() {
            Node<Float>* top{stack.top()};
            stack.pop();
            if(not top->is_leaf()) {
                stack.push(top->right_child);
                stack.push(top->left_child);
            }
            return *this;
        }
        inline const Node<Float>* operator*() const {
            if(stack.empty())
                return nullptr;
            else
                return stack.top();
        }
        inline bool operator==(const Iterator& other) const {
            return **this == *other;
        }
    private:
        std::stack<Node<Float>*> stack;
    };
    static_assert(std::is_same_v<Float, typename LossType::Float>);
    BaseRegressionTree(const TreeConfig& config_informations):
            config{config_informations},
            nb_splitting_nodes{0},
            root{nullptr}, fitted{false}, prop_root_p0{0.},
            data{nullptr}, owns_dataset{false},
            nodes() {
    }
    ~BaseRegressionTree() {
        if(root != nullptr) {
            root->data = nullptr;
            root->parent = nullptr;  // just to be sure
            delete root;
        }
        if(owns_dataset and data != nullptr)
            delete data;
        data = nullptr;
    }

    inline size_t get_nb_splitting_nodes() const {
        return nb_splitting_nodes;
    }

    void fit(const Dataset<Float>& dataset) {
        auto start{std::chrono::system_clock::now()};
        if(config.bootstrap) {
            data = dataset.sample(
                config.bootstrap_frac * dataset.size(),
                config.bootstrap_replacement
            );
            owns_dataset = true;
        } else {
            data = &dataset;
            owns_dataset = false;
        }
        is_categorical = Array<bool>(data->nb_features());
        for(size_t j{0}; j < is_categorical.size(); ++j)
            is_categorical[j] = data->is_categorical(j);
        // TODO: bootstraping
        if(data->is_weighted())
            prop_root_p0 = weighted_prop_false(data->get_p(), data->get_w());
        else
            prop_root_p0 = mean<bool, Float>(data->get_p());
        // TODO: handle fairness epsilon
        if(config.exact_splits) {
            for(size_t j{0}; j < data->nb_features(); ++j) {
                if(not is_categorical[j])
                    continue;
                auto nb_modalities{data->get_nb_unique_modalities(j)};
                constexpr size_t max_nb_modalities{20};
                if(nb_modalities > max_nb_modalities)
                    throw std::runtime_error(
                        std::to_string(nb_modalities) + " is too much for feature "
                        + std::to_string(j) + " (expected < "
                        + std::to_string(max_nb_modalities) + ")"
                    );
            }
        }
        if(config.split_type == NodeSelector::DEPTH_FIRST) {
            if constexpr(Loss::CanBeDepthFirst<LossType>::value) {
                if(config.normalized_dloss) {
                    build_tree<NodeSelector::DEPTH_FIRST, true>(*data);
                } else {
                    build_tree<NodeSelector::DEPTH_FIRST, false>(*data);
                }
            } else {
                throw std::runtime_error("Cannot be depth first");
            }
        } else if(config.split_type == NodeSelector::BEST_FIRST) {
            if constexpr(Loss::CanBeBestFirst<LossType>::value) {
                if(config.normalized_dloss) {
                    build_tree<NodeSelector::BEST_FIRST, true>(*data);
                } else {
                    build_tree<NodeSelector::BEST_FIRST, false>(*data);
                }
            } else {
                throw std::runtime_error("Cannot be best first");
            }
        } else
            throw std::runtime_error("Unknown NodeSelector");
        fitted = true;
        auto end{std::chrono::system_clock::now()};
        std::chrono::duration<double> elapsed{end-start};
        if(config.verbose) {
            std::cout << "Time elapsed: " << elapsed << "\n";
            std::cout << nb_splitting_nodes << " nodes\n";
        }
        nb_features = data->nb_features();
        if(owns_dataset) {
            delete data;
            data = nullptr;
        }
        for(const Node<Float>* node : *this) {
            if(node->parent != nullptr and node->data != nullptr) {
                delete node->data;
            }
            const_cast<Node<Float>*>(node)->data = nullptr;
        }
    }

    Array<Float> predict(const Array<Float>& X) const {
        Array<Float> ret(X.size() / nb_features);
        predict(X, ret);
        return ret;
    }

    void predict(const Array<Float>& X, Array<Float>& out) const {
        assert(root != nullptr);
        size_t n{out.size()};
        assert(n * nb_features == X.size());
        for(size_t i{0}; i < n; ++i) {
            out[i] = _predict(X, i, n);
        }
    }

    inline Float _predict(const Array<Float>& X, size_t i, size_t n) const {
        Node<Float>* node{root};
        while(not node->is_leaf()) {
            auto j{node->feature_idx};
            if(is_categorical[j]) {
                auto modality{static_cast<int>(X[j*n + i])};
                if(node->left_modalities & modality)
                    node = node->left_child;
                else if(node->right_modalities & modality)
                    node = node->right_child;
                else
                    break;
            } else {
                if(X[j*n + i] < node->threshold)
                    node = node->left_child;
                else
                    node = node->right_child;
            }
        }
        return node->mean_y;
    }

    inline const Node<Float>* get_root() const {
        return root;
    }

    inline const std::vector<Node<Float>*>& get_internal_nodes() const {
        return nodes;
    }

    inline Iterator begin() const {
        return root;
    }

    inline Iterator end() const {
        return nullptr;
    }

    void get_feature_importance(Float* ptr) const {
        Array<Float> view(ptr, nb_features, false);
        get_feature_importance(view);
    }

    void get_feature_importance(Array<Float>& importances) const {
        for(const Node<Float>* node : *this) {
            if(node->is_leaf())
                continue;
            importances[node->feature_idx] += node->dloss;
        }
        Float sum{Cart::sum(importances)};
        for(size_t j{0}; j < importances.size(); ++j)
            importances[j] /= sum;
    }
    Array<Float> get_feature_importance() const {
        Array<Float> ret(nb_features);
        get_feature_importance(ret);
        return ret;
    }
protected:
    const TreeConfig& config;
    size_t nb_splitting_nodes;
    Node<Float>* root;
    bool fitted;
    Float prop_root_p0;
    const Dataset<Float>* data;
    Array<bool> is_categorical;
    bool owns_dataset;
    size_t nb_features{0};

    std::vector<Node<Float>*> nodes;

    Float compute_loss(const Array<Float>& y) const {
        LossType loss;
        loss.augment(y);
        return loss();
    }

    Float compute_loss(const Array<Float>& y, const Array<Float>& w) const {
        LossType loss;
        loss.augment(y, w);
        return loss();
    }

    template <NodeSelector selector, bool normalized_dloss=true>
    void build_tree(const Dataset<Float>& dataset) {
        Splitter<Float, LossType, selector, normalized_dloss> splitter(dataset, config);
        do {
            Node<Float>* node{splitter.split()};
            if(node == nullptr)
                break;
            nodes.push_back(node);
            if(config.verbose) {
                std::cout << std::string(3*node->depth, ' ')
                    << "Node (" << node->id << "), Depth: " << node->depth
                    << ", Feature: " << node->feature_idx
                    << ", Threshold: " << node->threshold
                    << ", DLoss: " << std::setprecision(15) << node->dloss
                    << ", Loss: " << std::setprecision(15) << node->loss
                    << ", Mean_value: " << node->mean_y
                    << ", N=" << node->nb_observations
                    << '\n';
            }
            if(root == nullptr)
                root = node;
            ++nb_splitting_nodes;
        } while(nb_splitting_nodes < config.interaction_depth);
    }
};

template <typename Float, typename LossType>
requires(
    std::derived_from<LossType, Loss::NodeBasedLoss<Float, LossType>>
)
class NodeBasedRegressionTree final : public BaseRegressionTree<Float, LossType> {
};

}  // Cart::Regression::
}  // Cart::

#endif  // CART_TREE_HPP
