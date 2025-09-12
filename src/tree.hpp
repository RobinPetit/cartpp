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
            config{config_informations}, // data{nullptr},
            nb_splitting_nodes{0},
            root{nullptr}, fitted{false}, prop_root_p0{0.}, nodes() {
    }
    ~BaseRegressionTree() {
        if(root != nullptr) {
            root->data = nullptr;
            root->parent = nullptr;  // just to be sure
            delete root;
        }
    }

    inline size_t get_nb_splitting_nodes() const {
        return nb_splitting_nodes;
    }

    void fit(const Dataset<Float>& dataset) {
        auto start{std::chrono::system_clock::now()};
        // data = dataset;
        // TODO: bootstraping
        if(dataset.is_weighted())
            prop_root_p0 = weighted_prop_false(dataset.get_p(), dataset.get_w());
        else
            prop_root_p0 = mean<bool, Float>(dataset.get_p());
        // TODO: handle fairness epsilon
        if(config.exact_splits) {
            for(size_t j{0}; j < dataset.nb_features(); ++j) {
                if(not dataset.is_categorical(j))
                    continue;
                auto nb_modalities{dataset.get_nb_unique_modalities(j)};
                if(nb_modalities > 20)
                    throw std::runtime_error(
                        std::to_string(nb_modalities) + " is too much for feature "
                        + std::to_string(j)
                    );
            }
        }
        if(config.split_type == NodeSelector::DEPTH_FIRST) {
            if constexpr(Loss::CanBeDepthFirst<LossType>::value)
                build_tree<NodeSelector::DEPTH_FIRST>(dataset);
            else
                throw std::runtime_error("Cannot be depth first");
        } else if(config.split_type == NodeSelector::BEST_FIRST) {
            if constexpr(Loss::CanBeBestFirst<LossType>::value)
                build_tree<NodeSelector::BEST_FIRST>(dataset);
            else
                throw std::runtime_error("Cannot be best first");
        } else
            throw std::runtime_error("Unknown NodeSelector");
        fitted = true;
        auto end{std::chrono::system_clock::now()};
        std::chrono::duration<double> elapsed{end-start};
        std::cout << "Time elapsed: " << elapsed << "\n";
        std::cout << nb_splitting_nodes << " nodes\n";
    }

    Array<Float> predict(const Array<Float>& X) const {
        size_t nb_features{root->data->nb_features()};
        Array<Float> ret(X.size() / nb_features);
        predict(X, ret);
        return ret;
    }

    void predict(const Array<Float>& X, Array<Float>& out) const {
        size_t nb_features{root->data->nb_features()};
        size_t n{out.size()};
        assert(n * nb_features == X.size());
        for(size_t i{0}; i < n; ++i) {
            _predict(X.view(i*nb_features, (i+1)*nb_features), out[i]);
        }
    }

    inline void _predict(const Array<Float> x, Float& ret) const {
        Node<Float>* node{root};
        while(not node->is_leaf()) {
            auto j{node->feature_idx};
            if(root->data->is_categorical(j)) {
                auto modality{static_cast<int>(x[j])};
                if(node->left_modalities & modality)
                    node = node->left_child;
                else if(node->right_modalities & modality)
                    node = node->right_child;
                else
                    break;
            } else {
                if(x[j] < node->threshold)
                    node = node->left_child;
                else
                    node = node->right_child;
            }
        }
        ret = node->mean_y;
    }

    const std::vector<Node<Float>*>& get_internal_nodes() const {
        std::cout << "Returning a vector of size " << nodes.size() << "\n";
        return nodes;
    }
    inline Iterator begin() const {
        return root;
    }
    inline Iterator end() const {
        return nullptr;
    }
    void get_feature_importance(Float* ptr) const {
        Array<Float> view(ptr, root->data->nb_features(), false);
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
        Array<Float> ret(root->data->nb_features());
        get_feature_importance(ret);
        return ret;
    }
protected:
    const TreeConfig& config;
    size_t nb_splitting_nodes;
    Node<Float>* root;
    bool fitted;
    Float prop_root_p0;

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

    template <NodeSelector selector>
    void build_tree(const Dataset<Float>& dataset) {
        Splitter<Float, LossType, selector> splitter(dataset, config);
        do {
            Node<Float>* node{splitter.split()};
            if(node == nullptr)
                break;
            nodes.push_back(node);
            std::cout << std::string(3*node->depth, ' ')
                      << "Node (" << node->id << "), Depth: " << node->depth
                      << ", Feature: " << node->feature_idx
                      << ", Threshold: " << node->threshold
                      << ", DLoss: " << std::setprecision(15) << node->dloss
                      << ", Loss: " << std::setprecision(15) << node->loss
                      << ", Mean_value: " << node->mean_y
                      << ", N=" << node->nb_observations
                      << '\n';
            if(root == nullptr)
                root = node;
            ++nb_splitting_nodes;
        } while(nb_splitting_nodes < config.interaction_depth);
    }

    template <bool weighted>
    SplitChoice<Float> find_best_split(
            const Dataset<Float>* dataset, Float precomputed_loss) {
        Array<bool> usable(dataset->nb_features());
        for(size_t j{0}; j < usable.size(); ++j)
            usable[j] = dataset->not_all_equal(j);
        auto covariates{where(usable)};
        // TODO: subsample features for RF
        Float current_loss;
        if(std::isinf(precomputed_loss)) {
            if constexpr(weighted)
                current_loss = compute_loss(dataset->get_y(), dataset->get_w());
            else
                current_loss = compute_loss(dataset->get_y());
        } else
            current_loss = precomputed_loss;

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
