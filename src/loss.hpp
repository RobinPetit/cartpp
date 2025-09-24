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
#include "config.hpp"
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

    NodeBasedLoss(const TreeConfig&):
            NodeBasedLoss() {
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

    inline Float weighted_size() const {
        return sum_of_weights;
    }

    static inline Float get(const TreeConfig& config, const Array<Float>& ys) {
        LossType loss(config);
        loss.augment(ys);
        return loss;
    }
    static inline Float get(
            const TreeConfig& config,
            const Array<Float>& ys, const Array<Float>& ws) {
        LossType loss(config);
        loss.augment(ys, ws);
        return loss;
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
    using typename ParentLoss::Float;
#define END_OF_DEFINITION };

DEFINE_NODE_LOSS(MeanSquaredError)
public:
    MeanSquaredError():
            ParentLoss(), unweighted_sum{0},
            weighted_sum_squares{0} {
    }
    MeanSquaredError(const TreeConfig&):
            MeanSquaredError() {
    }
    ~MeanSquaredError() = default;
protected:
    // \sum_i y_i
    Float unweighted_sum;
    // \sum_i w_i y_i²
    Float weighted_sum_squares;

    inline Float compute() const override final {
        Float mu{weighted_sum / sum_of_weights};
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

namespace impl {
// Propagate CRTP
template <std::floating_point FloatType, class LossType>
class _NonNegativeIntegerLoss : public NodeBasedLoss<FloatType, LossType> {
private:
    typedef NodeBasedLoss<FloatType, LossType> ParentLoss;
    friend class NodeBasedLoss<FloatType, LossType>;
public:
    using typename ParentLoss::Float;
    _NonNegativeIntegerLoss():
            ParentLoss(),
            max_y{0}, sum_wi_when_y(16, 0.), unweighted_sum{0.} {
    }
    _NonNegativeIntegerLoss(const TreeConfig&):
            _NonNegativeIntegerLoss() {
    }
protected:
    using ParentLoss::sum_of_weights;
    using ParentLoss::weighted_sum;
    size_t max_y;
    std::vector<Float> sum_wi_when_y;
    // \sum_i y_i
    Float unweighted_sum;

    inline void _augment(const Array<Float>& ys) override final {
        _update_max_y(ys);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1, 0.);
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] += 1;
            unweighted_sum += ys[i];
        }
        weighted_sum = unweighted_sum;
        sum_of_weights += ys.size();
    }
    inline void _augment(const Array<Float>& ys,
                         const Array<Float>& ws) override final {
        _update_max_y(ys);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1, 0.);
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] += ws[i];
            unweighted_sum += ys[i];
            weighted_sum += ys[i]*ws[i];
            sum_of_weights += ws[i];
        }
    }
    inline void _augment(const LossType& other_loss) override final {
        max_y = std::max(max_y, other_loss.max_y);
        if(max_y >= sum_wi_when_y.size())
            sum_wi_when_y.resize(max_y+1, 0.);
        for(size_t y{0}; y <= other_loss.max_y; ++y)
            sum_wi_when_y[y] += other_loss.sum_wi_when_y[y];
        unweighted_sum += other_loss.unweighted_sum;
        weighted_sum += other_loss.weighted_sum;
        sum_of_weights += other_loss.sum_of_weights;
    }
    inline void _diminish(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] -= 1;
            unweighted_sum -= ys[i];
        }
        weighted_sum = unweighted_sum;
        sum_of_weights -= ys.size();
    }
    inline void _diminish(const Array<Float>& ys,
                          const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            sum_wi_when_y[static_cast<int>(ys[i])] -= ws[i];
            unweighted_sum -= ys[i];
            weighted_sum -= ys[i]*ws[i];
            sum_of_weights -= ws[i];
        }
    }
    inline void _diminish(const LossType& other_loss) override final {
        for(size_t y{0}; y <= other_loss.max_y; ++y)
            sum_wi_when_y[y] -= other_loss.sum_wi_when_y[y];
        unweighted_sum -= other_loss.unweighted_sum;
        weighted_sum -= other_loss.unweighted_sum;
        sum_of_weights -= other_loss.sum_of_weights;
    }
    inline void _update_max_y(const Array<Float>& ys) {
        auto max_y_in_sample{static_cast<size_t>(
            *std::max_element(ys.begin(), ys.end())
        )};
        if(max_y_in_sample > max_y) [[unlikely]]
            max_y = max_y_in_sample;
    }
};
}  // Cart::Loss::impl

template <std::floating_point FloatType>
class PoissonDeviance final : public impl::_NonNegativeIntegerLoss<
                                FloatType,
                                PoissonDeviance<FloatType>
                        > {
private:
    typedef impl::_NonNegativeIntegerLoss<
        FloatType,
        PoissonDeviance<FloatType>
    > ParentLoss;
    using ParentLoss::weighted_sum;
    using ParentLoss::sum_of_weights;
    using ParentLoss::sum_wi_when_y;
    using ParentLoss::max_y;
public:
    using typename ParentLoss::Float;
    friend class NodeBasedLoss<Float, PoissonDeviance<Float>>;
    friend ParentLoss;
    PoissonDeviance():
            ParentLoss() {
    }
    PoissonDeviance(const TreeConfig&):
            PoissonDeviance() {
    }
    ~PoissonDeviance() = default;
protected:
    inline Float compute() const override final {
        if(weighted_sum == 0) [[unlikely]]
            return 0;
        Float mu{weighted_sum / sum_of_weights};
        Float ret{sum_wi_when_y[0]*mu};
        for(size_t y{1}; y <= max_y; ++y) {
            ret += sum_wi_when_y[y]*(y*std::log(y / mu) + mu - y);
        }
        return 2 * ret / sum_of_weights;
    }
};

template <std::floating_point FloatType>
class NegativeBinomialDeviance final : public impl::_NonNegativeIntegerLoss<
                                            FloatType,
                                            NegativeBinomialDeviance<FloatType>
                                > {
private:
    typedef impl::_NonNegativeIntegerLoss<
        FloatType, NegativeBinomialDeviance<FloatType>
    > ParentLoss;
    using ParentLoss::weighted_sum;
    using ParentLoss::sum_of_weights;
    using ParentLoss::sum_wi_when_y;
    using ParentLoss::max_y;
public:
    using typename ParentLoss::Float;
    friend class NodeBasedLoss<Float, NegativeBinomialDeviance<Float>>;
    friend ParentLoss;
    NegativeBinomialDeviance():
            ParentLoss() {
    }
    NegativeBinomialDeviance(const TreeConfig& config):
            ParentLoss(config),
            alpha{static_cast<Float>(config._params._nb.alpha)} {
        assert(alpha > 0);
    }
    ~NegativeBinomialDeviance() = default;
protected:
    Float alpha;
    inline Float compute() const override final {
        if(weighted_sum == 0) [[unlikely]]
            return 0;
        assert(sum_of_weights > 0);
        Float mu{weighted_sum / sum_of_weights};
        Float ret{sum_wi_when_y[0] * std::log(1 + alpha*mu)};
        for(size_t y{1}; y <= max_y; ++y) {
            Float tmp{std::log((1 + alpha*mu) / (1 + alpha*y))};
            tmp += y*std::log((y*(1 + alpha*mu)) / (mu*(1 + alpha*y)));
            ret += sum_wi_when_y[y] * tmp;
            assert(std::isfinite(sum_wi_when_y[y]));
            assert(std::isfinite(tmp));
        }
        assert(std::isfinite(ret));
        if(alpha != 1.) {
            std::cout << "Alpha: " << alpha << "\n";
            assert(false);
        }
        return 2 * ret / (sum_of_weights * alpha);
    }
};

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
    virtual Float  _evaluate(
            const Array<Float>& y, size_t idx) const = 0;
    virtual Float  _evaluate(
            const Array<Float>& y, const Array<Float>& w, size_t idx) const = 0;
    // For categorical splits
    virtual Float _evaluate(uint64_t mask) = 0;
    virtual Float _evaluate(uint64_t mask,
            std::tuple<Float, Float, Float, Float>&) = 0;
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
    inline Float evaluate(const Array<Float>& y, const Array<Float>& w, size_t idx) const {
        return self._evaluate(y, w, idx);
    }
    inline Float evaluate(uint64_t mask) const {
        return self._evaluate(mask);
    }
    inline Float evaluate(
            uint64_t mask, std::tuple<Float , Float, Float, Float>& res) const {
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
    typedef TreeBasedLoss<
        FloatType,
        LorenzCurveError<FloatType, allow_crossing>
    > ParentLoss;
public:
    using typename ParentLoss::Float;
    LorenzCurveError(const Dataset<Float>& data):
            dataset{data}, curve() {
    }

    struct QuantileFunctionEntry {
        const Node<Float>* node;
        Float N;
        Float pred;
    };

    class LorenzCurve final {
    public:
        LorenzCurve() = default;
        LorenzCurve(const Node<Float>* root):
                quantiles(),
                sum_of_weights{root->sum_of_weights},
                Ey{root->pred} {
            quantiles.emplace_back(nullptr, Float(0.), Float(0.));
            quantiles.emplace_back(root, sum_of_weights, Ey);
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
            CARTPP_UNREACHABLE
            return gamma;
        }

        inline void split_node(const Node<Float>* node) {
            auto left{node->left_child};
            auto right{node->right_child};
            split_node(
                node,
                left,  left->sum_of_weights,  left->pred,
                right, right->sum_of_weights, right->pred
            );
        }

        inline void split_node(
                const Node<Float>* node,
                Float left, Float pred_left,
                Float right, Float pred_right) {
            split_node(
                node,
                nullptr, left, pred_left,
                nullptr, right, pred_right
            );
        }

        inline void split_node(
                const Node<Float>* node,
                const Node<Float>* left_node, Float left, Float pred_left,
                const Node<Float>* right_node, Float right, Float pred_right) {
            quantiles.emplace_back(
                nullptr, Float(0.), std::numeric_limits<Float>::infinity()
            );
            auto it{std::find_if(
                quantiles.begin() + 1,
                quantiles.end() - 1,
                [node](const auto& entry) -> bool {
                    return entry.node == node;
                }
            )};
            QuantileFunctionEntry small_entry{left_node, left, pred_left};
            QuantileFunctionEntry big_entry{right_node, right, pred_right};
            if(big_entry.pred < small_entry.pred)
                std::swap(small_entry, big_entry);

            if(small_entry.pred >= (it-1)->pred) [[unlikely]] {
                *it = small_entry;
            } else {
                _insert(quantiles.begin()+1, it+1, small_entry);
            }
            if(big_entry.pred > (quantiles.end()-1)->pred) [[unlikely]] {
                quantiles.back() = big_entry;
            } else {
                _insert(++it, quantiles.end(), big_entry);
            }
            precomputed = false;
        }

        inline auto begin() const {
            return get_lc().begin();
        }
        inline auto end() const {
            return get_lc().end();
        }

        inline size_t size() const {
            return get_lc().size();
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
            return static_cast<Float>(.5) * ret;
        }

        inline bool crosses(const LorenzCurve& other, Float eps=1e-8) const {
            for(auto [gamma, LC_gamma] : *this)
                if(LC_gamma > other(gamma) + eps)
                    return true;
            return false;
        }

        inline size_t count_crossings(const LorenzCurve& other, Float eps=1e-8) const {
            size_t ret{0};
            for(auto [gamma, LC_gamma] : *this)
                if(LC_gamma > other(gamma) + eps)
                    ++ret;
            return ret;
        }
    private:
        std::vector<QuantileFunctionEntry> quantiles;
        std::vector<Coord<Float>> _precomputed_lc;
        bool precomputed{false};
        Float sum_of_weights;
        Float Ey;

        inline const std::vector<Coord<Float>>& get_lc() const {
            if(not precomputed) [[unlikely]]
                _compute();
            return _precomputed_lc;
        }

        inline void _compute() const {
            auto& lc{const_cast<std::vector<Coord<Float>>&>(_precomputed_lc)};
            lc.clear();
            lc.reserve(quantiles.size());
            lc.emplace_back(Float(0.), Float(0.));
            Float last_pred{0};
            for(auto it{quantiles.begin()}; it != quantiles.end(); ++it) {
                if(it->pred == last_pred) [[unlikely]] {
                    lc.back().first  += it->N;
                    lc.back().second += it->pred*it->N;
                } else {
                    last_pred = it->pred;
                    lc.emplace_back(
                        lc.back().first  + it->N,
                        lc.back().second + it->pred*it->N
                    );
                }
            }
            for(auto& [gamma, LC_gamma] : lc) {
                gamma /= sum_of_weights;
                LC_gamma /= sum_of_weights*Ey;
            }
            const_cast<bool&>(precomputed) = true;
        }

        template <typename It>
        inline void _insert(
                It first, It last,
                QuantileFunctionEntry const& entry) {
            auto it{std::find_if(
                first, last,
                [&entry](const QuantileFunctionEntry& x) -> bool {
                    return x.pred >= entry.pred;
                }
            )};
            std::shift_right(it, last, 1);
            *it = entry;
        }
    };
protected:
    friend class TreeBasedLoss<Float, LorenzCurveError<Float, allow_crossing>>;

    const Dataset<Float>& dataset;

    const Node<Float>* current_node{nullptr};
    Float left_sum{0};
    Float left_sum_of_weights{0};
    Float right_sum_of_weights{0};
    Float right_sum{0};
    size_t last_idx{0};
    size_t nb_modalities{0};
    Float total_size{0};
    Float total_sum{0};

    LorenzCurve curve;
    std::vector<std::pair<Float, Float>> _mod_N_pred;

    static inline Float _evaluate(const LorenzCurve& curve) {
        // std::cout << "Evaluating on LC:\n";
        // for(auto [gamma, LC] : curve)
        //     std::cout << "(" << gamma << ", " << LC << ")   ";
        // std::cout << '\n';
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
        if(current_node->data->is_weighted()) {
            right_sum = weighted_sum(y, w);
            right_sum_of_weights = sum(w);
        } else {
            right_sum = sum(y);
            right_sum_of_weights = static_cast<Float>(y.size());
        }
        left_sum_of_weights = 0;
        last_idx = 0;
        total_size = 0;
        total_sum = 0;
        // std::cout << "new feature " << j << "\n";
        if(current_node->data->is_categorical(j)) {
            auto [values, counts] = unique(Xj);
            auto sumcounts{cumsum<size_t>(counts)};
            nb_modalities = counts.size();
            _mod_N_pred.clear();
            for(size_t k{0}; k < nb_modalities; ++k) {
                size_t base_idx{(k == 0) ? 0 : sumcounts[k-1]};
                size_t idx{sumcounts[k]};
                if(current_node->data->is_weighted()) {
                    auto ws{w.view(base_idx, idx)};
                    _mod_N_pred.emplace_back(
                        sum(ws),
                        weighted_sum(y.view(base_idx, idx), ws)
                    );
                } else {
                    _mod_N_pred.emplace_back(
                        static_cast<Float>(idx - base_idx),
                        sum(y.view(base_idx, idx))
                    );
                }
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
            const Array<Float>& w,
            size_t idx) const override final {
        LorenzCurve splitted_curve(curve);
        auto ws{w.view(last_idx, idx)};
        auto diff{weighted_sum<Float>(y.view(last_idx, idx), ws)};
        auto _this{const_cast<LorenzCurveError<Float, allow_crossing>*>(this)};
        _this->last_idx = idx;
        _this->left_sum  += diff;
        _this->right_sum -= diff;
        auto diff_weights{sum(ws)};
        _this->left_sum_of_weights  += diff_weights;
        _this->right_sum_of_weights -= diff_weights;
        splitted_curve.split_node(
            current_node,
            left_sum_of_weights, left_sum / left_sum_of_weights,
            right_sum_of_weights, right_sum / right_sum_of_weights
        );
        if constexpr(not allow_crossing) {
            if(splitted_curve.crosses(curve))
                return -std::numeric_limits<Float>::infinity();
        }
        return _evaluate(splitted_curve);
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
        _this->left_sum_of_weights  = static_cast<Float>(idx);
        _this->right_sum_of_weights = static_cast<Float>(y.size() - idx);
        splitted_curve.split_node(
            current_node,
            left_sum_of_weights, left_sum / left_sum_of_weights,
            right_sum_of_weights, right_sum / right_sum_of_weights
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
            std::tuple<Float, Float, Float, Float>& res) override final {
        return _evaluate(mask, &res);
    }

    Float _evaluate(
            uint64_t mask,
            std::tuple<Float, Float, Float, Float>* res) {
        LorenzCurve splitted_curve(curve);
        left_sum = 0;
        right_sum = 0;
        Float left_size{0};
        Float right_size{0};
        for(size_t mod_idx{0}; mod_idx < nb_modalities; ++mod_idx) {
            if(mask & (1ull << mod_idx)) {
                left_sum  += _mod_N_pred[mod_idx].second;
                left_size += _mod_N_pred[mod_idx].first;
            }
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
};

template <std::floating_point Float>
using NonCrossingLorenzCurveError = LorenzCurveError<Float, false>;
template <std::floating_point Float>
using CrossingLorenzCurveError = LorenzCurveError<Float, true>;

template <std::floating_point Float>
static inline auto _consecutive_lcs(const std::vector<Node<Float>*>& nodes) {
    typename LorenzCurveError<Float>::LorenzCurve lc(nodes.front());
    std::vector<typename Cart::Loss::LorenzCurveError<Float>::LorenzCurve> ret;
    ret.push_back(lc);
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
