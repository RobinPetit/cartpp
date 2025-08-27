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
requires(
    std::derived_from<LossType, Loss::NodeBasedLoss<Float, LossType>>
)
class BaseRegressionTree {
public:
    BaseRegressionTree(const TreeConfig& config_informations):
            config{config_informations}, // data{nullptr},
            nb_splitting_nodes{0},
            root{nullptr}, fitted{false}, prop_root_p0{0.} {
    }
    ~BaseRegressionTree() {
        if(root != nullptr) {
            root->data = nullptr;
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
        // std::cout << "fit()\n";
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
        if(config.split_type == NodeSelector::DEPTH_FIRST)
            build_tree<NodeSelector::DEPTH_FIRST>(dataset);
        else if(config.split_type == NodeSelector::BEST_FIRST)
            build_tree<NodeSelector::BEST_FIRST>(dataset);
        else
            throw std::runtime_error("Unknown NodeSelector");
        fitted = true;
        auto end{std::chrono::system_clock::now()};
        auto elapsed{end-start};
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)/1e9 << '\n';
        std::cout << nb_splitting_nodes << " nodes\n";
    }
protected:
    const TreeConfig& config;
    // std::shared_ptr<const Dataset<Float>> data;
    size_t nb_splitting_nodes;
    Node<Float>* root;
    bool fitted;
    Float prop_root_p0;

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
        // ...
        do {
            Node<Float>* node{splitter.split()};
            if(node == nullptr) {
                // std::cout << "EARLY STOP\n";
                break;
            }
            for(size_t _{0}; _ <= node->depth; ++_)
                std::cout << "  ";
            std::cout << "Node (" << node->id << "), Depth: " << node->depth
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
class NodeBasedRegressionTree final : public BaseRegressionTree<Float, LossType> {
};

// template <typename Float>
// class TreeBasedRegressionTree final : public BaseRegressionTree<Float> {
// public:
//     TreeBasedRegressionTree(const TreeConfig& config_informations):
//             BaseRegressionTree<Float>(config_informations) {
//         // TODO: complete
//     }
//
// private:
// };

}  // Cart::Regression::
}  // Cart::

#endif  // CART_TREE_HPP
