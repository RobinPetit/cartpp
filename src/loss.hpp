#ifndef CART_LOSS_HPP
#define CART_LOSS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "array.hpp"
#include "node.hpp"

namespace Cart {
namespace Loss {

// A whole bunch of Curiously Recurring Template Pattern for static polymorphism.
// (c.f. https://en.cppreference.com/w/cpp/language/crtp.html)
// Also the syntax is wordy but we need to be C++-20 compliant
// ~(let's hope this compiles on MSVC++ at some point in time...)~
// IT DOES! Who knew? Way to go Bill

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
    virtual Float  _evaluate() const = 0;  // TODO: remove?
    // For numerical splits
    virtual Float  _evaluate(const Array<Float>& y, size_t idx) const = 0;
    // For categorical splits
    virtual Float _evaluate(uint64_t mask) = 0;
    virtual Float _evaluate(uint64_t mask,
            std::tuple<size_t, Float, size_t, Float>&) = 0;
    virtual void _set_root(Node<Float>* node) = 0;
    virtual void _new_node(const Node<Float>* node) = 0;
    virtual void _new_feature(size_t j) = 0;
public:
    TreeBasedLoss():
            self{static_cast<LossType&>(*this)} {}
    ~TreeBasedLoss() = default;

    inline void new_node(const Node<Float>* node) {
        self._new_node(node);
    }
    inline void new_feature(size_t j) {
        self._new_feature(j);
    }

    inline Float evaluate(const Array<Float>& y, size_t idx) const {
        return self._evaluate(y, idx);
    }
    inline Float evaluate(uint64_t mask) const {
        return self._evaluate(mask);
    }
    inline Float evaluate(
            uint64_t mask, std::tuple<size_t, Float, size_t, Float>& res) const {
        return self._evaluate(mask, res);
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

template <std::floating_point Float>
using Coord = std::pair<Float, Float>;

template <std::floating_point FloatType, bool allow_crossing=true>
class LorenzCurveError final : public TreeBasedLoss<
                                    FloatType,
                                    LorenzCurveError<FloatType, allow_crossing>
                          > {
    typedef TreeBasedLoss<FloatType, LorenzCurveError<FloatType, allow_crossing>> ParentLoss;
public:
    using typename ParentLoss::Float;
    LorenzCurveError(const Dataset<Float>& data): dataset{data}, curve() {
    }

    struct _Entry {
        const Node<Float>* node;
        size_t N;
        Float pred;
    };

    class LorenzCurve final {
    public:
        class Iterator {
        private:
            using _BaseIterator = std::vector<_Entry>::const_iterator;
        public:
            using difference_type = long;
            using value_type = Coord<Float>;
            using pointer = value_type*;
            using reference = value_type&;
            using iterator_category = std::input_iterator_tag;
            Iterator() = delete;
            Iterator(_BaseIterator iterator, size_t size, Float expected_value):
                    it{iterator},
                    n{0},
                    N{size},
                    LC_gamma{0},
                    Ey{expected_value} {
            }
            inline Iterator& operator++() {
                n += it->N;
                LC_gamma += it->N * it->pred;
                ++it;
                return *this;
            }
            inline Coord<Float> operator*() const {
                auto norm_N{static_cast<Float>(N)};
                auto norm_LC{static_cast<Float>(Ey*N)};
                return {
                    static_cast<Float>(n+it->N) / norm_N,
                    static_cast<Float>(LC_gamma + it->N*it->pred) / norm_LC
                };
            }
            inline bool operator==(const Iterator& other) const {
                return it == other.it;
            }
        private:
            _BaseIterator it;
            size_t n;
            size_t N;
            Float LC_gamma;
            Float Ey;
        };

        LorenzCurve() = default;
        LorenzCurve(const Node<Float>* root):
                quantiles(),
                N{root->nb_observations},
                Ey{
                    (root->data == nullptr)
                    ? root->mean_y
                    : mean<Float, Float>(root->data->get_y())
                } {
            quantiles.emplace_back(nullptr, 0, Float(0.));
            quantiles.emplace_back(root, N, Ey);
        }
        LorenzCurve(const LorenzCurve& other) = default;
        LorenzCurve(LorenzCurve&& other) = default;

        inline Float operator()(Float gamma) const {
            Float last_gamma{0};
            Float LC_last_gamma{0};
            for(auto [gamma_i, LC_gamma_i] : *this) {
                if(gamma <= gamma_i) {
                    auto dx{gamma_i - last_gamma};
                    auto dy{LC_gamma_i - LC_last_gamma};
                    return LC_gamma_i + dy/dx * (gamma - last_gamma);
                }
                last_gamma = gamma_i;
                LC_last_gamma = LC_gamma_i;
            }
            // [[unreachable]]
            return gamma;
        }

        void split_node(const Node<Float>* node) {
            auto it{std::find_if(
                quantiles.begin(),
                quantiles.end(),
                [node](const auto& entry) {
                    return entry.node == node;
                }
            )};
            assert(it != quantiles.end());
            assert(node->left_child != nullptr);
            assert(node->right_child != nullptr);
            *it = {
                node->left_child,
                node->left_child->nb_observations,
                node->left_child->mean_y
            };
            quantiles.emplace_back(
                node->right_child,
                node->right_child->nb_observations,
                node->right_child->mean_y
            );
            _sort(quantiles);
        }
        void split_node(
                const Node<Float>* node,
                size_t left, Float pred_left,
                size_t right, Float pred_right) {
            auto it{std::find_if(
                quantiles.begin(),
                quantiles.end(),
                [node](const auto& entry) {
                    return entry.node == node;
                }
            )};
            assert(it != quantiles.end());
            assert(node->left_child == nullptr);
            assert(node->right_child == nullptr);
            *it = {nullptr, left, pred_left};
            quantiles.emplace_back(nullptr, right, pred_right);
            _sort(quantiles);
        }
        inline Iterator begin() const {
            return Iterator(quantiles.begin(), N, Ey);
        }
        inline Iterator end() const {
            return Iterator(quantiles.end(), N, Ey);
        }

        operator std::vector<Coord<Float>>() const {
            return std::vector<Coord<Float>>(begin(), end());
        }

        inline Float area() const {
            Float ret{0};
            Float last_LC{0};
            Float last_gamma{0};
            for(auto [gamma, LC_gamma] : *this) {
                ret += (gamma - last_gamma) * (LC_gamma + last_LC);
                last_LC = LC_gamma;
                last_gamma = gamma;
            }
            return .5 * ret;
        }

        // TODO: optimize this: profile says we spend ~80% of the time in crosses
        inline bool crosses(const LorenzCurve& other, Float eps=1e-8) {
            for(auto [gamma, LC_gamma] : *this)
                if(LC_gamma > other(gamma) + eps)
                    return true;
            return false;
        }
    private:
        std::vector<_Entry> quantiles;
        size_t N;
        Float Ey;

        static inline void _sort(std::vector<_Entry>& array) {
            std::sort(
                array.begin(), array.end(),
                [](const auto& a, const auto& b) -> bool {
                    return a.pred < b.pred;
                }
            );
        }
    };
protected:
    friend class TreeBasedLoss<Float, LorenzCurveError<Float, allow_crossing>>;
    using ParentLoss::self;

    const Dataset<Float>& dataset;

    const Node<Float>* current_node{nullptr};
    Float left_sum{0};
    Float right_sum{0};
    size_t last_idx{0};
    size_t nb_modalities{0};
    size_t total_size{0};
    Float total_sum{0};

    LorenzCurve curve;
    std::vector<std::pair<size_t, Float>> _mod_N_pred;

    static inline Float _evaluate(const LorenzCurve& curve) {
        return static_cast<Float>(1) - 2*curve.area();
    }
    inline Float _evaluate() const override final {
        return _evaluate(curve);
    }

    inline void _set_root(Node<Float>* node) override final {
        new(&curve) LorenzCurve(node);
    }

    virtual inline void _add_expanded_node(const Node<Float>* node) override final {
        curve.split_node(node);
    }

    virtual inline void _new_node(const Node<Float>* node) override final {
        current_node = node;
    }

    virtual inline void _new_feature(size_t j) override final {
        auto const& [Xj, y, p, w, indices] = current_node->data->sorted_Xypw(j);
        left_sum = 0;
        right_sum = sum(y);
        last_idx = 0;
        total_size = 0;
        total_sum = 0;
        if(current_node->data->is_categorical(j)) {
            auto [values, counts] = unique(Xj);
            auto sumcounts{cumsum<size_t>(counts)};
            nb_modalities = counts.size();
            _mod_N_pred.clear();
            for(size_t k{0}; k < nb_modalities; ++k) {
                size_t base_idx{(k == 0) ? 0 : sumcounts[k-1]};
                size_t idx{sumcounts[k]};
                _mod_N_pred.emplace_back(
                    idx-base_idx,
                    sum(y.view(base_idx, idx))
                );
                total_size += _mod_N_pred.back().first;
                total_sum += _mod_N_pred.back().second;
            }
        } else {
            _mod_N_pred.clear();
            nb_modalities = 0;
        }
    }

    virtual inline Float _evaluate(
            const Array<Float>& y,
            size_t idx) const override final {
        LorenzCurve splitted_curve(curve);
        auto diff{sum(y.view(last_idx, idx))};
        auto _this{const_cast<LorenzCurveError<Float, allow_crossing>*>(this)};
        _this->last_idx = idx;
        _this->left_sum += diff;
        _this->right_sum -= diff;
        splitted_curve.split_node(
            current_node,
            idx, left_sum / idx,
            y.size() - idx, right_sum / (y.size() - idx)
        );
        if constexpr(not allow_crossing) {
            if(splitted_curve.crosses(curve))
                return -std::numeric_limits<Float>::infinity();
        }
        return _evaluate(splitted_curve);
    }
    virtual inline Float _evaluate(uint64_t mask) override final {
        return _evaluate(mask, nullptr);
    }
    virtual inline Float _evaluate(
            uint64_t mask,
            std::tuple<size_t, Float, size_t, Float>& res) override final {
        return _evaluate(mask, &res);
    }
    Float _evaluate(
            uint64_t mask,
            std::tuple<size_t, Float, size_t, Float>* res) {
        LorenzCurve splitted_curve(curve);
        left_sum = 0;
        right_sum = 0;
        size_t left_size{0};
        size_t right_size{0};
        for(size_t mod_idx{0}; mod_idx < nb_modalities; ++mod_idx) {
            if(mask & (1ull << mod_idx)) {
                left_sum += _mod_N_pred[mod_idx].second;
                left_size += _mod_N_pred[mod_idx].first;
            } /*else {
                right_sum += _mod_N_pred[mod_idx].second;
                right_size += _mod_N_pred[mod_idx].first;
            }*/
        }
        right_size = total_size - left_size;
        right_sum = total_sum - left_sum;
        splitted_curve.split_node(
            current_node,
            left_size, left_sum / left_size,
            right_size, right_sum / right_size
        );
        if(res != nullptr) [[likely]] {
            std::get<0>(*res) = left_size;
            std::get<1>(*res) = left_sum / left_size;
            std::get<2>(*res) = right_size;
            std::get<3>(*res) = right_sum / right_size;
        }
        return _evaluate(splitted_curve);
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
        // [[unreachable]]
        return gamma;
    }
};

template <std::floating_point Float>
using NonCrossingLorenzCurveError = LorenzCurveError<Float, false>;
template <std::floating_point Float>
using CrossingLorenzCurveError = LorenzCurveError<Float, true>;

template <std::floating_point Float>
static inline auto _consecutive_lcs(const std::vector<Node<Float>*>& nodes) {
    typename LorenzCurveError<Float>::LorenzCurve lc(nodes.front());
    std::vector<std::vector<Coord<Float>>> ret;
    ret.emplace_back(lc);
    for(const Node<Float>* node : nodes) {
        lc.split_node(node);
        ret.push_back(lc);
    }
    return ret;
}

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
