#ifndef CART_SPLITTER_HPP
#define CART_SPLITTER_HPP

#include <concepts>
#include <cstdint>
#include <limits>
#include <queue>
#include <stack>
#include <stdexcept>

#include "array.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "node.hpp"
#include "config.hpp"

namespace Cart {

template <typename Float>
struct SplitChoice {
    bool valid;
    bool is_categorical;
    int feature_idx;
    Float dloss;
    Float left_loss, right_loss;
    const Dataset<Float>* left_data;
    const Dataset<Float>* right_data;
    Node<Float>* node;

    // For numerical splits
    Float threshold;

    // For categorical splits
    uint64_t left_modalities;
    uint64_t right_modalities;
};

template <std::floating_point Float>
struct NodeDlossComparatorMaxHeap {
    typedef Node<Float> * const Ptr;
    inline bool operator()(const Ptr lhs, const Ptr rhs) {
        return lhs->dloss < rhs->dloss;
    }
};

template <std::floating_point Float, NodeSelector type>
struct Container {
};
template <std::floating_point Float>
struct Container<Float, NodeSelector::BEST_FIRST> {
    typedef std::priority_queue<
        Node<Float>*,
        std::vector<Node<Float>*>,
        NodeDlossComparatorMaxHeap<Float>
    > type;
};
template <std::floating_point Float>
struct Container<Float, NodeSelector::DEPTH_FIRST> {
    typedef std::stack<Node<Float>*> type;
};
template <std::floating_point Float, NodeSelector selector>
using ContainerType = Container<Float, selector>::type;

template <std::floating_point Float, typename LossType, NodeSelector selector, bool normalized_dloss>
class Splitter;

namespace impl {
template <
    std::floating_point Float,
    typename LossType
>
class Splitter {
public:
    template <bool weighted, bool normalized_dloss>
    static inline bool find_best_split(
            const TreeConfig& config,
            const Node<Float>* node, size_t j,
            Float current_loss, SplitChoice<Float>& best_split) {
        if(node->data->is_categorical(j)) {
            return find_best_split_categorical<weighted, normalized_dloss>(
                config, node, j, current_loss, best_split
            );
        } else {
            return find_best_split_numerical<weighted, normalized_dloss>(
                config, node, j, current_loss, best_split
            );
        }
    }

    template <bool weighted, bool normalized_dloss>
    static bool find_best_split_categorical(
            const TreeConfig& config,
            const Node<Float>* node, size_t j,
            Float current_loss, SplitChoice<Float>& best_split) {
        auto data{node->data};
        const auto& [Xj, y, p, w, indices] = data->sorted_Xypw(j);
        if(Xj[0] == Xj[Xj.size()-1])
            return false;
        auto [values, counts] = unique(Xj);
        auto sumcounts{cumsum<size_t>(counts)};
        size_t nb_modalities{counts.size()};
        if(nb_modalities > 64)
            throw std::runtime_error("");
        std::vector<LossType> losses(nb_modalities);
        for(size_t k{0}; k < nb_modalities; ++k) {
            size_t base_idx{(k == 0) ? 0 : sumcounts[k-1]};
            size_t idx{sumcounts[k]};
            if constexpr(weighted) {
                losses[k].augment(
                    y.view(base_idx, idx),
                    w.view(base_idx, idx)
                );
            } else {
                losses[k].augment(y.view(base_idx, idx));
            }
        }
        Float best_dloss{best_split.dloss};
        Float best_loss_left{0};
        Float best_loss_right{0};
        uint64_t best_mask{0};
        auto max_mask{1ull << (nb_modalities-1)};
        for(uint64_t _mask{1ull}; _mask < max_mask; ++_mask) {
            LossType loss_left;
            LossType loss_right;
            auto mask{_mask};
            for(size_t mod_idx{0}; mod_idx < nb_modalities; ++mod_idx) {
                if(mask & (1ull << mod_idx))
                    loss_left.augment(losses[mod_idx]);
                else
                    loss_right.augment(losses[mod_idx]);
            }
            if(loss_left.size() < config.minobs or loss_right.size() < config.minobs)
                continue;
            Float dloss{
                data->size() * current_loss
                - (loss_left.size() * loss_left + loss_right.size() * loss_right)
            };
            if constexpr(normalized_dloss)
                dloss /= data->size();
            if(dloss > best_dloss) {
                best_dloss = dloss;
                best_mask = _mask;
                best_loss_left = loss_left;
                best_loss_right = loss_right;
            }
        }
        if(best_dloss > best_split.dloss) {
            best_split.valid = true;
            best_split.is_categorical = true;
            best_split.feature_idx = static_cast<int>(j);
            best_split.dloss = best_dloss;
            best_split.left_loss = best_loss_left;
            best_split.right_loss = best_loss_right;
            best_split.left_modalities = 0;
            best_split.right_modalities = 0;
            Array<bool> go_left(data->size(), false);
            Array<bool> go_right(data->size(), false);
            for(size_t k{0}; k < nb_modalities; ++k) {
                uint64_t flag{1};
                size_t base_idx{(k == 0) ? 0 : sumcounts[k-1]};
                size_t end_idx{sumcounts[k]};
                flag <<= static_cast<int>(values[k]);
                if(best_mask & (1ull << k)) {
                    best_split.left_modalities |= flag;
                    go_left.view(base_idx, end_idx).assign(true);
                } else {
                    best_split.right_modalities |= flag;
                    go_right.view(base_idx, end_idx).assign(true);
                }
            }
            if(best_split.left_data != nullptr)
                delete best_split.left_data;
            if(best_split.right_data != nullptr)
                delete best_split.right_data;
            best_split.left_data = data->at(indices[go_left]);
            best_split.right_data = data->at(indices[go_right]);
            best_split.threshold = -1;
        }
        return true;
    }

    template <bool weighted, bool normalized_dloss>
    static bool find_best_split_numerical(
            const TreeConfig& config,
            const Node<Float>* node, size_t j,
            Float current_loss, SplitChoice<Float>& best_split) {
        auto data{node->data};
        if(data->size() < 2*config.minobs)
            return false;
        const auto& [Xj, y, p, w, indices] = data->sorted_Xypw(j);
        if(Xj[0] == Xj[Xj.size()-1])
            return false;
        Array<bool> left_mask(data->size(), false);
        Array<bool> right_mask(data->size(), true);
        LossType left_loss;
        LossType right_loss;
        if constexpr(weighted) {
            right_loss.augment(y, w);
        } else {
            right_loss.augment(y);
        }
        size_t idx{0};
        Float best_dloss{best_split.dloss};
        Float best_threshold{std::numeric_limits<Float>::infinity()};
        Float best_loss_left{0.};
        Float best_loss_right{0.};
        size_t best_splitting_idx{0};

        while(true) {
            Float prev_value{Xj[idx]};
            size_t base_idx{idx};
            while(idx < Xj.size() and Xj[idx] == prev_value) {
                left_mask[idx] = true;
                right_mask[idx] = false;
                ++idx;
            }
            if(idx == Xj.size())
                break;
            if constexpr(weighted) {
                auto sub_y{y.view(base_idx, idx)};
                auto sub_w{w.view(base_idx, idx)};
                left_loss.augment(sub_y, sub_w);
                right_loss.diminish(sub_y, sub_w);
            } else {
                auto sub_y{y.view(base_idx, idx)};
                left_loss.augment(sub_y);
                right_loss.diminish(sub_y);
            }
            if(idx < config.minobs)
                continue;
            if(data->size() - idx < config.minobs)
                break;
            assert(idx > 0);
            assert(idx < y.size());
            Float dloss{
                data->size() * current_loss
                - (idx*left_loss + (data->size()-idx)*right_loss)
            };
            if constexpr(normalized_dloss)
                dloss /= data->size();
            if(dloss > best_dloss) {
                best_dloss = dloss;
                best_threshold = (prev_value + Xj[idx]) / static_cast<Float>(2);
                best_splitting_idx = idx;
                best_loss_left = left_loss;
                best_loss_right = right_loss;
            }
        }
        if(best_dloss > best_split.dloss) {
            best_split.valid = true;
            best_split.is_categorical = false;
            best_split.feature_idx = static_cast<int>(j);
            best_split.threshold = best_threshold;
            best_split.dloss = best_dloss;
            best_split.left_loss = best_loss_left;
            best_split.right_loss = best_loss_right;
            if(best_split.left_data != nullptr)
                delete best_split.left_data;
            if(best_split.right_data != nullptr)
                delete best_split.right_data;
            best_split.left_data = data->at(
                indices.view(0, best_splitting_idx)
            );
            best_split.right_data = data->at(
                indices.view(best_splitting_idx, indices.size())
            );
        }
        return true;
    }

    static inline void _expand_node(
            Node<Float>* parent,
            const SplitChoice<Float>& split) {
        assert(split.valid);
        assert(split.left_loss == LossType::get(split.left_data->get_y()));
        assert(split.right_loss == LossType::get(split.right_data->get_y()));
        parent->left_child = new Node<Float>(
            -1, parent->depth+1, split.left_data, parent
        );
        parent->right_child = new Node<Float>(
            -1, parent->depth+1, split.right_data, parent
        );
        parent->dloss = split.dloss;
        parent->feature_idx = split.feature_idx;
        parent->threshold = split.threshold;
        parent->left_child->loss = split.left_loss;
        parent->right_child->loss = split.right_loss;
    }

private:
};
}

/* Node-based loss specialization */
template <
    std::floating_point Float,
    Loss::_NodeBasedLoss<Float> LossType,
    NodeSelector selector,
    bool normalized_dloss
>
class Splitter<Float, LossType, selector, normalized_dloss> final {
public:
    Splitter() = delete;
    Splitter(const Dataset<Float>& data, const TreeConfig& config_):
            node_counter{0}, dataset{data}, container(), config{config_} {
        Node<Float>* root{new Node<Float>(node_counter++, 0, &data)};
        if(data.is_weighted())
            root->loss = LossType::get(data.get_y(), data.get_w());
        else
            root->loss = LossType::get(data.get_y());
        expand(root);
    }

    ~Splitter() {
        while(not container.empty()) {
            Node<Float>* node{container.top()};
            container.pop();
            delete node->left_child;
            delete node->right_child;
            node->left_child = node->right_child = nullptr;
        }
    }

    Node<Float>* split() {
        if(container.empty())
            return nullptr;
        Node<Float>* ret{container.top()};
        container.pop();
        expand(ret->right_child);
        expand(ret->left_child);
        return ret;
    }
private:
    size_t node_counter;
    const Dataset<Float>& dataset;
    ContainerType<Float, selector> container;
    const TreeConfig& config;

    using Implementation = impl::Splitter<Float, LossType>;

    void expand(Node<Float>* node) {
        if(node->depth == config.max_depth)
            return;
        SplitChoice<Float> best_split;
        best_split.left_data = best_split.right_data = nullptr;
        best_split.valid = false;
        best_split.dloss = 0;
        Array<bool> usable(node->data->nb_features(), false);
        for(size_t j{0}; j < usable.size(); ++j) {
            auto const& [Xj, y, p, w, indices] = node->data->sorted_Xypw(j);
            usable[j] = Xj[0] != Xj[Xj.size()-1];
        }
        Array<size_t> features{where(usable)};
        if(config.nb_covariates != 0 and features.size() > config.nb_covariates)
            features = Random::choice(features, config.nb_covariates, false);
        for(size_t j : features) {
            if(node->data->is_weighted()) {
                Implementation::template find_best_split<true, normalized_dloss>(
                    config, node, j, node->loss, best_split
                );
            } else {
                Implementation::template find_best_split<false, normalized_dloss>(
                    config, node, j, node->loss, best_split
                );
            }
        }
        if(best_split.valid) {
            node->feature_idx = best_split.feature_idx;
            node->dloss = best_split.dloss;
            node->threshold = best_split.threshold;
            Implementation::_expand_node(node, best_split);
            node->id = node_counter++;
            if(node->parent != nullptr) {
                delete node->data;
                node->data = nullptr;
            }
            node->left_modalities = best_split.left_modalities;
            node->right_modalities = best_split.right_modalities;
            container.emplace(node);
        }
    }
};

/* Tree-based loss specialization */
// NOTE: it makes no sense to build it depth first, hence the specialization
template <
    std::floating_point Float,
    Loss::_TreeBasedLoss<Float> LossType,
    bool normalized_dloss
>
class Splitter<Float, LossType, NodeSelector::BEST_FIRST, normalized_dloss> final {
public:
    Splitter() = delete;
    Splitter(const Dataset<Float>& data, const TreeConfig& config_):
            node_counter{0}, dataset{data}, container(), config{config_},
            loss(data) {
        Node<Float>* root{new Node<Float>(node_counter++, 0, &data)};
        loss.set_root(root);
        container.push_back(root);
    }

    ~Splitter() {
        container.clear();
    }

    Node<Float>* split() {
        SplitChoice<Float> split{_find_split()};
        if(not split.valid)
            return nullptr;
        Node<Float>* node{split.node};
        node->feature_idx = split.feature_idx;
        node->threshold = split.threshold;
        node->loss = static_cast<Float>(-1);
        node->dloss = split.dloss;
        node->loss = loss;
        node->left_modalities = split.left_modalities;
        node->right_modalities = split.right_modalities;
        node->left_child = new Node<Float>(
            node_counter++, node->depth+1, split.left_data, node
        );
        node->right_child = new Node<Float>(
            node_counter++, node->depth+1, split.right_data, node
        );
        container.erase(
            std::find_if(
                container.begin(), container.end(),
                [node](const auto p) { return p == node; }
            )
        );
        container.emplace_back(node->left_child);
        container.emplace_back(node->right_child);
        loss.add_expanded_node(node);
        return node;
    }
private:
    size_t node_counter;
    const Dataset<Float>& dataset;
    std::vector<Node<Float>*> container;
    const TreeConfig& config;
    LossType loss;

    using Implementation = impl::Splitter<Float, LossType>;

    SplitChoice<Float> _find_split() {
        SplitChoice<Float> best_split;
        best_split.left_data = best_split.right_data = nullptr;
        best_split.valid = false;
        best_split.dloss = 0;
        best_split.node = nullptr;
        for(auto node : container) {
            if(node->depth == config.max_depth)
                continue;
            loss.new_node(node);
            Array<bool> usable(dataset.nb_features(), false);
            for(size_t j{0}; j < usable.size(); ++j) {
                auto const& [Xj, y, p, w, indices] = node->data->sorted_Xypw(j);
                usable[j] = Xj[0] != Xj[Xj.size()-1];
            }
            Array<size_t> features{where(usable)};
            if(config.nb_covariates != 0 and features.size() > config.nb_covariates)
                features = Random::choice(features, config.nb_covariates, false);
            for(size_t j : features)
                find_best_split(config, node, j, best_split);
        }
        return best_split;
    }

    void find_best_split(
            const TreeConfig& config, Node<Float>* node,
            size_t j, SplitChoice<Float>& best_split) {
        if(node->data->is_categorical(j)) {
            find_best_split_categorical(config, node, j, best_split);
        } else {
            find_best_split_numerical(config, node, j, best_split);
        }
    }
    void find_best_split_numerical(
            const TreeConfig& config, Node<Float>* node,
            size_t j, SplitChoice<Float>& best_split) {
        const Dataset<Float>* data{node->data};
        if(data->size() < 2*config.minobs)
            return;
        const auto& [Xj, y, p, w, indices] = data->sorted_Xypw(j);
        loss.new_feature(j);
        if(Xj[0] == Xj[Xj.size()-1])
            return;
        Float best_threshold{std::numeric_limits<Float>::infinity()};
        Float best_dloss{0};
        size_t idx{0};
        size_t best_splitting_idx{0};
        Float current_loss{loss};
        while(true) {
            Float prev_value{Xj[idx]};
            while(idx < Xj.size() and Xj[idx] == prev_value) {
                ++idx;
            }
            if(idx == Xj.size())
                break;
            if(idx < config.minobs)
                continue;
            if(data->size() - idx < config.minobs)
                break;
            Float new_loss{loss.evaluate(y, idx)};
            Float dloss{new_loss - current_loss};
            if(dloss > best_dloss) {
                best_threshold = (Xj[idx] + prev_value) / 2;
                best_dloss = dloss;
                best_splitting_idx = idx;
            }
        }
        if(best_dloss > best_split.dloss) {
            best_split.valid = true;
            best_split.is_categorical = false;
            best_split.node = node;
            best_split.threshold = best_threshold;
            best_split.feature_idx = static_cast<int>(j);
            best_split.dloss = best_dloss;
            if(best_split.left_data != nullptr)
                delete best_split.left_data;
            if(best_split.right_data != nullptr)
                delete best_split.right_data;
            best_split.left_data = node->data->at(
                indices.view(0, best_splitting_idx)
            );
            best_split.right_data = node->data->at(
                indices.view(best_splitting_idx, y.size())
            );
        }
    }

    void find_best_split_categorical(
            const TreeConfig& config, Node<Float>* node,
            size_t j, SplitChoice<Float>& best_split) {
        auto data{node->data};
        assert(data != nullptr);
        const auto& [Xj, y, p, w, indices] = data->sorted_Xypw(j);
        loss.new_feature(j);
        if(Xj[0] == Xj[Xj.size()-1])
            return;
        auto [values, counts] = unique(Xj);
        auto sumcounts{cumsum<size_t>(counts)};
        size_t nb_modalities{counts.size()};
        if(nb_modalities > 64)
            throw std::runtime_error("");
        Float best_dloss{best_split.dloss};
        uint64_t best_mask{0};
        Float current_loss{loss};
        auto max_mask{1ull << (nb_modalities-1)};
        for(uint64_t _mask{1ull}; _mask < max_mask; ++_mask) {
            auto mask{_mask};
            std::tuple<size_t, Float, size_t, Float> _split_res;
            Float new_loss{loss.evaluate(mask, _split_res)};
            if(std::get<0>(_split_res) < config.minobs or
               std::get<2>(_split_res) < config.minobs)
                continue;
            Float dloss{new_loss - current_loss};
            if(dloss > best_dloss) {
                best_dloss = dloss;
                best_mask = _mask;
            }
        }
        if(best_dloss > best_split.dloss) {
            best_split.valid = true;
            best_split.is_categorical = true;
            best_split.node = node;
            best_split.feature_idx = static_cast<int>(j);
            best_split.dloss = best_dloss;
            best_split.left_modalities = 0;
            best_split.right_modalities = 0;
            Array<bool> go_left(data->size(), false);
            Array<bool> go_right(data->size(), false);
            for(size_t k{0}; k < nb_modalities; ++k) {
                uint64_t flag{1};
                size_t base_idx{(k == 0) ? 0 : sumcounts[k-1]};
                size_t end_idx{sumcounts[k]};
                flag <<= static_cast<int>(values[k]);
                if(best_mask & (1ull << k)) {
                    best_split.left_modalities |= flag;
                    go_left.view(base_idx, end_idx).assign(true);
                } else {
                    best_split.right_modalities |= flag;
                    go_right.view(base_idx, end_idx).assign(true);
                }
            }
            if(best_split.left_data != nullptr)
                delete best_split.left_data;
            if(best_split.right_data != nullptr)
                delete best_split.right_data;
            best_split.left_data = data->at(indices[go_left]);
            best_split.right_data = data->at(indices[go_right]);
            best_split.threshold = -1;
        }
    }
};

}

#endif
