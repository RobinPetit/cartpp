#ifndef CART_LOSS_HPP
#define CART_LOSS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>

#include "array.hpp"
#include "node.hpp"

namespace Cart {
namespace Loss {

// A whole bunch of Curiously Recurring Template Pattern (c.f. https://en.cppreference.com/w/cpp/language/crtp.html)
// for static polymorphism.
// Also the syntax is wordy but we need to be C++-20 compliant
// (let's hope this compiles on MSVC++ at some point in time...)

template <std::floating_point Float, class LossType>
class NodeBasedLoss {
protected:
    // Precomputed loss value
    Float value;
    // Number of observations
    size_t n;
    // Whether or not `value` is up to date
    bool precomputed;
    // \sum_i w_i
    Float sum_of_weights;
    // \sum_i w_i y_i
    Float weighted_sum;

    LossType& self;

    virtual inline Float compute() const = 0;
    virtual inline void _augment(const Array<Float>& ys) = 0;
    virtual inline void _augment(const Array<Float>& ys, const Array<Float>& ws) = 0;
    virtual inline void _augment(const LossType&) = 0;
    virtual inline void _diminish(const Array<Float>& ys) = 0;
    virtual inline void _diminish(const Array<Float>& ys, const Array<Float>& ws) = 0;
    virtual inline void _diminish(const LossType&) = 0;

public:
    NodeBasedLoss():
            value{0}, n{0}, precomputed{false},
            sum_of_weights{0}, weighted_sum{0},
            self{static_cast<LossType&>(*this)} {
    }

    ~NodeBasedLoss() = default;

    inline Float evaluate() {
        if(not precomputed) {
            // Use self for static polymorphism
            value = self.compute();
            precomputed = true;
        }
        return value;
    }
    inline Float operator()() {
        return evaluate();
    }
    inline operator Float() {
        return evaluate();
    }
    inline void augment(const Array<Float>& ys) {
        n += ys.size();
        self._augment(ys);
        precomputed = false;
    }
    inline void augment(const Array<Float>& ys, const Array<Float>& ws) {
        n += ys.size();
        self._augment(ys, ws);
        precomputed = false;
    }
    inline void augment(const LossType& other_loss) {
        n += other_loss.n;
        self._augment(other_loss);
        precomputed = false;
    }
    inline void diminish(const Array<Float>& ys) {
        n -= ys.size();
        self._diminish(ys);
        precomputed = false;
    }
    inline void diminish(const Array<Float>& ys, const Array<Float>& ws) {
        n -= ys.size();
        self._diminish(ys, ws);
        precomputed = false;
    }
    inline void diminish(const LossType& other_loss) {
        n -= other_loss.n;
        self._diminish(other_loss);
        precomputed = false;
    }
    inline size_t size() const {
        return n;
    }
};

#define DEFINE_NODE_LOSS(NAME) \
template <std::floating_point FloatType> \
class NAME final : public NodeBasedLoss<FloatType, NAME<FloatType>> { \
public: \
    typedef FloatType Float; \
    static Float get(const Array<Float>& ys) { \
        NAME<Float> loss; \
        loss.augment(ys); \
        return loss; \
    } \
    static Float get(const Array<Float>& ys, const Array<Float>& ws) { \
        NAME<Float> loss; \
        loss.augment(ys, ws); \
        return loss; \
    } \
private: \
    typedef NodeBasedLoss<Float, NAME<Float>> ParentLoss; \
protected: \
    friend class NodeBasedLoss<Float, NAME<Float>>; \
    using ParentLoss::value; \
    using ParentLoss::n; \
    using ParentLoss::precomputed; \
    using ParentLoss::sum_of_weights; \
    using ParentLoss::weighted_sum; \
    using ParentLoss::self;
#define END_OF_DEFINITION };

DEFINE_NODE_LOSS(MeanSquaredError)
public:
    MeanSquaredError():
        ParentLoss(), unweighted_sum{0},
        weighted_sum_squares{0} {
    }
    ~MeanSquaredError() = default;
protected:
    // \sum_i y_i
    Float unweighted_sum;
    // \sum_i w_i y_i²
    Float weighted_sum_squares;

    inline Float compute() const override final {
        Float mu{unweighted_sum / n};
        Float ret{
            weighted_sum_squares
            - 2 * mu * weighted_sum
        };
        ret /= sum_of_weights;
        ret += mu*mu;
        return ret;
    }
    inline void _augment(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            unweighted_sum += ys[i];
            weighted_sum += ys[i];
            weighted_sum_squares += ys[i]*ys[i];
        }
        sum_of_weights += ys.size();
    }
    inline void _augment(const Array<Float>& ys,
                         const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            unweighted_sum += ys[i];
            weighted_sum += ys[i]*ws[i];
            weighted_sum_squares += ys[i]*ys[i]*ws[i];
            sum_of_weights += ws[i];
        }
    }
    inline void _augment(const MeanSquaredError<Float>& other_loss) {
        unweighted_sum += other_loss.unweighted_sum;
        weighted_sum += other_loss.weighted_sum;
        weighted_sum_squares += other_loss.weighted_sum_squares;
        sum_of_weights += other_loss.sum_of_weights;
    }
    inline void _diminish(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            unweighted_sum -= ys[i];
            weighted_sum -= ys[i];
            weighted_sum_squares -= ys[i]*ys[i];
        }
        sum_of_weights -= ys.size();
    }
    inline void _diminish(const Array<Float>& ys,
                          const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            unweighted_sum -= ys[i];
            weighted_sum -= ys[i]*ws[i];
            weighted_sum_squares -= ys[i]*ys[i]*ws[i];
            sum_of_weights -= ws[i];
        }
    }
    inline void _diminish(const MeanSquaredError<Float>& other_loss) {
        unweighted_sum -= other_loss.unweighted_sum;
        weighted_sum -= other_loss.weighted_sum;
        weighted_sum_squares -= other_loss.weighted_sum_squares;
        sum_of_weights -= other_loss.sum_of_weights;
    }
END_OF_DEFINITION

DEFINE_NODE_LOSS(PoissonDeviance)
public:
    PoissonDeviance():
        ParentLoss(), max_y{0}, sum_wi_when_y(16, 0.), unweighted_sum{0.} {
    }
    ~PoissonDeviance() = default;
protected:
    size_t max_y;
    std::vector<Float> sum_wi_when_y;
    // \sum_i y_i
    Float unweighted_sum;

    inline Float compute() const override final {
        assert(n > 0);
        Float mu{unweighted_sum / static_cast<Float>(n)};
        if(mu == 0)
            return 0;
        Float ret{sum_wi_when_y[0]*mu};
        Float sum_of_weights{sum_wi_when_y[0]};
        for(size_t y{1}; y <= max_y; ++y) {
            sum_of_weights += sum_wi_when_y[y];
            ret += sum_wi_when_y[y]*(y*std::log(y / mu) + mu - y);
        }
        return 2 * ret / sum_of_weights;
    }
    inline void _augment(const Array<Float>& ys) override final {
        _update_max_y(ys);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1, 0.);
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] += 1;
            unweighted_sum += ys[i];
        }
    }
    inline void _augment(const Array<Float>& ys,
                         const Array<Float>& ws) override final {
        _update_max_y(ys);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1);
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] += ws[i];
            unweighted_sum += ys[i];
        }
    }
    inline void _augment(const PoissonDeviance<Float>& other_loss) override final {
        max_y = std::max(max_y, other_loss.max_y);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1, 0.);
        for(size_t y{0}; y <= other_loss.max_y; ++y)
            sum_wi_when_y[y] += other_loss.sum_wi_when_y[y];
        unweighted_sum += other_loss.unweighted_sum;
        precomputed = false;
    }
    inline void _diminish(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] -= 1;
            unweighted_sum -= ys[i];
        }
    }
    inline void _diminish(const Array<Float>& ys,
                          const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] -= ws[i];
            unweighted_sum -= ys[i];
        }
    }
    inline void _diminish(const PoissonDeviance<Float>& other_loss) override final {
        for(size_t y{0}; y <= other_loss.max_y; ++y)
            sum_wi_when_y[y] -= other_loss.sum_wi_when_y[y];
        unweighted_sum -= other_loss.unweighted_sum;
    }
    inline void _update_max_y(const Array<Float>& ys) {
        auto max_y_in_sample{*std::max_element(ys.begin(), ys.end())};
        if(max_y_in_sample > max_y) [[unlikely]]
            max_y = max_y_in_sample;
    }
END_OF_DEFINITION

#undef DEFINE_NODE_LOSS
#undef END_OF_DEFINITION

template <std::floating_point Float, class LossType>
class TreeBasedLoss {
protected:
    LossType& self;

    virtual void _add_expanded_node(const Node<Float>* node) = 0;
    virtual Float  _evaluate() = 0;
    virtual Float  _evaluate(Node<Float>* node, const Array<Float>& y, size_t idx) = 0;
public:
    TreeBasedLoss():
            self{static_cast<LossType&>(*this)} {}
    ~TreeBasedLoss() = default;

    inline Float evaluate() {
        return self._evaluate();
    }
    inline Float operator()() {
        return evaluate();
    }
    inline operator Float() {
        return evaluate();
    }

    inline void add_expanded_node(const Node<Float>* node) {
        self._add_expanded_node(node);
    }
};

template <std::floating_point Float>
class LorenzCurve final : public TreeBasedLoss<Float, LorenzCurve<Float>> {
public:
    LorenzCurve(const Dataset<Float>& data): dataset{data}, entries() {
    }
private:
    typedef TreeBasedLoss<Float, LorenzCurve<Float>> ParentLoss;
protected:
    friend class TreeBasedLoss<Float, LorenzCurve<Float>>;
    using ParentLoss::self;

    inline auto _find_it(Node<Float>* node) const {
        return std::find_if(
            entries.begin(), entries.end(),
            [node](const auto& entry) {
                return entry.node == node;
            }
        );
    }
    inline size_t _find_idx(Node<Float>* node) const {
        auto it{_find_it(node)};
        assert(it != entries.end());
        return it - entries.begin();
    }

    virtual inline void _add_expanded_node(Node<Float>* node) override final {
        auto idx{_find_idx(node)};
        entries[idx] = {
            node->left_child, node->left_child->size(), node->left_child->mean_y
        };
        entries.emplace_back(
            node->right_child, node->right_child->size(), node->right_child->mean_y
        );
        _sort(entries);
    }

    virtual inline Float _evaluate(
            Node<Float>* node, const Array<Float>& y, size_t idx) override final {
        Float new_area{0};
        std::vector<_Entry> tmp_entries(entries);
        tmp_entries.erase(_find_it(node));
        tmp_entries.emplace_back(nullptr, idx, mean(y.view(0, idx)));
        tmp_entries.emplace_back(nullptr, y.size()-idx, mean(y.view(idx, y.size())));
        _sort(tmp_entries);
    }

    struct _Entry {
        Node<Float>* node;
        size_t N;
        Float pred;
    };
    const Dataset<Float>& dataset;
    std::vector<_Entry> entries;

    static inline void _sort(std::vector<_Entry>& vec) {
        struct {
        public:
            inline bool operator()(_Entry const& a, _Entry const& b) const {
                return a.mean_y < b.mean_y;
            }
        } _order;
        std::sort(vec.begin(), vec.end(), _order);
    }
    static inline Float _area(const std::vector<_Entry>& vec) {
        return 0.;  // TODO
        // Float area{0};
        // Float last_y{0};
        // size_t last_N{0};
        // for(auto& [_, N, y]: vec) {
        //     base_area += (y - last_y) / (.N - last_N);
        //     last_y = entry.mean_y;
        //     last_N = entry.N;
        // }
        // base_area /= (2 * entries.back().mean_y * entries.back().N);
    }
};

template <typename LossType, typename Float=typename LossType::Float>
concept _NodeBasedLoss = requires {
    requires std::derived_from<LossType, NodeBasedLoss<Float, LossType>>;
};
template <typename LossType, typename Float=typename LossType::Float>
concept _TreeBasedLoss = requires {
    requires std::derived_from<LossType, TreeBasedLoss<Float, LossType>>;
};
template <typename LossType, typename Float=typename LossType::Float>
concept _Loss = requires {
    requires _NodeBasedLoss<LossType, Float>
          or _TreeBasedLoss<LossType, Float>;
};

}  // Cart::Loss::
}  // Cart::

#endif
