#ifndef CART_LOSS_HPP
#define CART_LOSS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>

#include "array.hpp"
#include "node.hpp"

namespace Cart {
namespace Loss {

// A whole bunch of Curiously Recurring Template Pattern for static polymorphism.
// (c.f. https://en.cppreference.com/w/cpp/language/crtp.html)
// Also the syntax is wordy but we need to be C++-20 compliant
// (let's hope this compiles on MSVC++ at some point in time...)

template <std::floating_point FloatType, class LossType>
class NodeBasedLoss {
public:
    typedef FloatType Float;
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
private: \
    typedef NodeBasedLoss<FloatType, NAME<FloatType>> ParentLoss; \
protected: \
    friend class NodeBasedLoss<FloatType, NAME<FloatType>>; \
    using ParentLoss::value; \
    using ParentLoss::n; \
    using ParentLoss::precomputed; \
    using ParentLoss::sum_of_weights; \
    using ParentLoss::weighted_sum; \
    using ParentLoss::self; \
public: \
    using typename ParentLoss::Float; \
    static Float get(const Array<Float>& ys) { \
        NAME<Float> loss; \
        loss.augment(ys); \
        return loss; \
    } \
    static Float get(const Array<Float>& ys, const Array<Float>& ws) { \
        NAME<Float> loss; \
        loss.augment(ys, ws); \
        return loss; \
    }
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
    inline void _augment(const MeanSquaredError<Float>& other_loss) override final {
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
    inline void _diminish(const MeanSquaredError<Float>& other_loss) override final {
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
        auto max_y_in_sample{static_cast<size_t>(
            *std::max_element(ys.begin(), ys.end())
        )};
        if(max_y_in_sample > max_y) [[unlikely]]
            max_y = max_y_in_sample;
    }
END_OF_DEFINITION

#undef DEFINE_NODE_LOSS
#undef END_OF_DEFINITION

template <std::floating_point FloatType, class LossType>
class TreeBasedLoss {
public:
    typedef FloatType Float;
protected:
    LossType& self;

    virtual void _add_expanded_node(const Node<Float>* node) = 0;
    virtual Float  _evaluate() const = 0;
    virtual Float  _evaluate(const Node<Float>* node, const Array<Float>& y, size_t idx) const = 0;
    virtual void _set_root(Node<Float>* node);
public:
    TreeBasedLoss():
            self{static_cast<LossType&>(*this)} {}
    ~TreeBasedLoss() = default;

    inline Float evaluate(
            const Node<Float>* node, const Array<Float>& y, size_t idx) const {
        return self._evaluate(node, y, idx);
    }
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

    inline void set_root(Node<Float>* node) {
        self._set_root(node);
    }
};

template <std::floating_point FloatType, bool allow_crossing=false>
class LorenzCurve final : public TreeBasedLoss<
                                    FloatType,
                                    LorenzCurve<FloatType, allow_crossing>
                          > {
    typedef TreeBasedLoss<FloatType, LorenzCurve<FloatType, allow_crossing>> ParentLoss;
public:
    using typename ParentLoss::Float;
    LorenzCurve(const Dataset<Float>& data): dataset{data}, entries() {
    }

    struct _Entry {
        Node<Float>* node;
        size_t N;
        Float pred;
    };
    static inline Float _area(const std::vector<_Entry>& vec) {
        auto lc{__compute_LC(vec)};
        Float ret{0};
        Float last_LC{0};
        Float last_gamma{0};
        for(auto [gamma, LC_gamma] : lc) {
            ret += (gamma - last_gamma) * (LC_gamma + last_LC);
            last_LC = LC_gamma;
            last_gamma = gamma;
        }
        return ret / 2;
    }
protected:
    friend class TreeBasedLoss<Float, LorenzCurve<Float, allow_crossing>>;
    using ParentLoss::self;

    static auto __compute_LC(const std::vector<_Entry>& vec) {
        size_t tot_N{0};
        for(auto [_, N, y] : vec)
            tot_N += N;
        std::vector<std::pair<Float, Float>> lc;
        lc.emplace_back(0, 0);
        size_t cumsum_N{0};
        for(auto [_, N, pi] : vec) {
            cumsum_N += N;
            auto gamma{cumsum_N / static_cast<Float>(tot_N)};
            lc.emplace_back(gamma, lc.back().second + (gamma - lc.back().first)*pi);
        }
        for(auto& pair : lc)
            pair.second /= lc.back().second;
        return lc;
    }

    const Dataset<Float>& dataset;
    std::vector<_Entry> entries;


    inline Float _evaluate() const override final {
        return .5 - _area(entries);
    }

    virtual inline void _set_root(Node<Float>* node) override final {
        _add_node(node);
    }

    inline auto _find_it(const Node<Float>* node) const {
        return _find_it(node, entries);
    }
    inline auto _find_it(
            const Node<Float>* node, const std::vector<_Entry>& array) const {
        return std::find_if(
            array.begin(), array.end(),
            [node](const auto& entry) {
                return entry.node == node;
            }
        );
    }
    inline size_t _find_idx(const Node<Float>* node) const {
        auto it{_find_it(node)};
        assert(it != entries.end());
        return it - entries.begin();
    }

    inline void _add_node(Node<Float>* node) {
        entries.emplace_back(node, node->nb_observations, node->mean_y);
    }
    virtual inline void _add_expanded_node(const Node<Float>* node) override final {
        auto idx{_find_idx(node)};
        entries[idx] = {
            node->left_child, node->left_child->nb_observations,
            node->left_child->mean_y
        };
        entries.emplace_back(
            node->right_child, node->right_child->nb_observations,
            node->right_child->mean_y
        );
        _sort(entries);
    }

    virtual inline Float _evaluate(
            const Node<Float>* node, const Array<Float>& y,
            size_t idx) const override final {
        std::vector<_Entry> tmp_entries(entries);
        if(tmp_entries.size() > 0) {
            auto it{_find_it(node, tmp_entries)};
            assert(it != tmp_entries.end());
            tmp_entries.erase(it);
        }
        auto yleft{y.view(0, idx)};
        auto yright{y.view(idx, y.size())};
        tmp_entries.emplace_back(nullptr, idx, mean(yleft));
        tmp_entries.emplace_back(nullptr, y.size()-idx, mean(yright));
        _sort(tmp_entries);
        if constexpr(not allow_crossing) {
            bool verbose{false};
            if(crosses(tmp_entries, verbose))
                return -std::numeric_limits<Float>::infinity();
        }
        return .5 - _area(tmp_entries);
    }

    static inline void _sort(std::vector<_Entry>& vec) {
        struct {
            inline bool operator()(_Entry const& a, _Entry const& b) const {
                return a.pred < b.pred;
            }
        } _order;
        std::sort(vec.begin(), vec.end(), _order);
    }

    static inline bool is_sorted(const std::vector<_Entry>& vec) {
        struct {
            inline bool operator()(_Entry const& a, _Entry const& b) const {
                return a.pred < b.pred;
            }
        } _order;
        return std::is_sorted(vec.begin(), vec.end(), _order);
    }

    static Float _evaluate_lc(
            const std::vector<std::pair<Float, Float>>& lc, Float gamma) {
        Float last_gamma{0};
        Float last_LC{0};
        for(auto [gamma_i, LC_gamma_i] : lc) {
            if(gamma <= gamma_i) {
                auto dx{gamma_i - last_gamma};
                auto dy{LC_gamma_i - last_LC};
                return LC_gamma_i + dy/dx * (gamma - last_gamma);
                last_gamma = gamma_i;
                last_LC = LC_gamma_i;
            }
        }
    }

    inline bool crosses(const std::vector<_Entry>& vec, bool verbose=false) const {
        auto new_lc{__compute_LC(vec)};
        auto curr_lc{__compute_LC(entries)};
        for(auto [gamma, LC_gamma] : new_lc) {
            if(_evaluate_lc(curr_lc, gamma) < LC_gamma - 1e-8)
                return true;
        }
        return false;
        // TODO: optimize
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

template <typename LossType>
struct CanBeDepthFirst {
    static constexpr bool value{true};
};

template <Loss::_TreeBasedLoss LossType>
struct CanBeDepthFirst<LossType> {
    static constexpr bool value{false};
};

template <Loss::_Loss LossType>
struct CanBeBestFirst {
    static constexpr bool value{true};
};


}  // Cart::Loss::
}  // Cart::

#endif
