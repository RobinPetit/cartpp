/**
 * @file loss.hpp
 *
 * @brief Implementation of the different losses.
 *
 * Heavily relies on [CRTP](https://en.cppreference.com/w/cpp/language/crtp.html).
 */
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

/**
 * @namespace Cart
 *
 * @brief Namespace containing everything that is defined in cartpp.
 */
namespace Cart {
/**
 * @namespace Cart::Loss
 *
 * In this namespace, one will find all the losses that are implemented.
 * See @file loss.hpp.
 */
namespace Loss {
/**
 * @brief Abstract wrapper for losses to be used during the construction of a
 * regression/classification tree.
 *
 * The idea to avoid having a method
 * `compute(const Array<Float>& y, const Array<Float>& yhat)`
 * (or more precisely `compute(const Array<Float>& y, Float yhat)` for CART)
 * to compute the loss but rather to keep an intermediate state that is updated
 * when needed, and from which the actual value can be efficiently computed.
 *
 * The class must implement some methods (including compute, augment and diminish)
 * that implement the required behaviour of each loss function and updated state.
 *
 * For all loss classes, we use the following notations:
 * \f{eqnarray*}{
 * W &=& \sum_{i=1}^nw_i \\
 * \hat\pi(y, w) &=& \frac{1}{W}\sum_{i=1}^nw_iy_i,
 * \f}
 * where \f$n\f$ is the number of observations added to the loss
 * (see size()).
 *
 * @tparam FloatType The (floating-point) type of data.
 * @tparam Float Alias of FloatType.
 * @tparam LossType Used for [CRTP](https://en.cppreference.com/w/cpp/language/crtp.html).
 */
template <std::floating_point FloatType, class LossType>
class NodeBasedLoss {
public:
    typedef FloatType Float;
protected:
    /// Precomputed loss value
    Float value;
    /// Number of observations
    size_t n;
    /// Whether or not `value` is up to date
    bool precomputed;
    /// \f$sum_i w_i\f$, i.e. \f$W\f$
    Float sum_of_weights;
    /// \f$\sum_i w_i y_i\f$, i.e. \f$W \cdot \pi(y, w)\f$.
    Float weighted_sum;

    /// Reference to `this` of the right type.
    /// This allows for static polymorphism and speed up runtime.
    LossType& self;

    /**
     * @brief Evaluate the loss on its current state.
     *
     * Use the attributes encoding the current state to compute the value of
     * the loss function.
     *
     * Should only be called by evaluate.
     */
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

    /**
     * @brief Constructor from TreeConfig.
     *
     * Uses information in TreeConfig::_params if needed (in child classes).
     * Simply an alias of NodeBasedLoss() in the base class.
     */
    NodeBasedLoss(const TreeConfig&):
            NodeBasedLoss() {
    }

    virtual ~NodeBasedLoss() = default;

    /**
     * @brief Evaluate the loss on its current state.
     *
     * Cache the value so that in two consecutive calls to evaluate()
     * only the first one will actually compute the value;
     * the second one will only access the cache.
     */
    inline Float evaluate() {
        if(not precomputed) [[likely]] {
            // Use self for static polymorphism
            value = self.compute();
            precomputed = true;
        }
        return value;
    }

    /**
     * @brief Evaluate the loss.
     *
     * Convenient alias of evaluate().
     */
    inline Float operator()() {
        return evaluate();
    }

    /**
     * @brief Evaluate the loss.
     *
     * Convenient alias of evaluate().
     */
    inline operator Float() {
        return evaluate();
    }

    /**
     * @brief Update the current state by adding some values.
     *
     * Add the values of `ys` to the considered values for the loss.
     *
     * @param ys The ground truth values to add to the current state.
     */
    inline void augment(const Array<Float>& ys) {
        n += ys.size();
        self._augment(ys);
        precomputed = false;
    }

    /**
     * @brief Update the current state by adding some values.
     *
     * This is a weighted version of augment(const Array<Float>&).
     *
     * @param ys The ground truth values to add to the current state.
     * @param ws The associated weights.
     */
    inline void augment(const Array<Float>& ys, const Array<Float>& ws) {
        n += ys.size();
        self._augment(ys, ws);
        precomputed = false;
    }

    /**
     * @brief Update the current state by adding the state of another same loss.
     *
     * In some cases (typically for categorical covariates), it might be
     * interesting to precompute the contribution to the loss function
     * only once in the beginning, and then combine different "sublosses"
     * with their precomputed state.
     *
     * @param other_loss The other loss to merge with `this`.
     */
    inline void augment(const LossType& other_loss) {
        n += other_loss.n;
        self._augment(other_loss);
        precomputed = false;
    }

    /**
     * @brief Update the current state by removing some values.
     *
     * This is the opposite operation of augment(const Array<Float>&).
     */
    inline void diminish(const Array<Float>& ys) {
        n -= ys.size();
        self._diminish(ys);
        precomputed = false;
    }

    /**
     * @brief Update the current state by removing some values.
     *
     * This is the opposite operation of augment(const Array<Float>&, const Array<Float>&).
     */
    inline void diminish(const Array<Float>& ys, const Array<Float>& ws) {
        n -= ys.size();
        self._diminish(ys, ws);
        precomputed = false;
    }

    /**
     * @brief Update the current state by removing the state of another loss.
     *
     * This is the opposite operation of augment(const LossType&).
     */
    inline void diminish(const LossType& other_loss) {
        n -= other_loss.n;
        self._diminish(other_loss);
        precomputed = false;
    }

    /**
     * @brief Get the number of observations seen in this state.
     */
    inline size_t size() const {
        return n;
    }

    /**
     * @brief Get the sum of the weights of the observations seen in this state.
     */
    inline Float weighted_size() const {
        return sum_of_weights;
    }

    /**
     * @brief Get the loss associated with given values.
     *
     * Typically equivalent to
     * ```
     * LossType loss(config);
     * loss.augment(ys);
     * return loss;
     * ```
     *
     * See NodeBasedLoss(const TreeConfig&) and augment(const Array<Float>&).
     *
     * @param config The config containing potential additional params.
     * @param ys The ground truth values to compute the loss on.
     */
    static inline Float get(const TreeConfig& config, const Array<Float>& ys) {
        LossType loss(config);
        loss.augment(ys);
        return loss;
    }

    /**
     * @brief Get the loss associated with given values.
     *
     * Weighted version of get(const TreeConfig&, const Array<Float>&).
     *
     */
    static inline Float get(
            const TreeConfig& config,
            const Array<Float>& ys, const Array<Float>& ws) {
        LossType loss(config);
        loss.augment(ys, ws);
        return loss;

    }

    /**
     * @brief Compute the weighted mean of the \f$y_i\f$'s.
     *
     * More precisely, compute \f$\hat\pi(y, w)\f$.
     */
    inline Float get_mu() const {
        return weighted_sum / sum_of_weights;
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

/**
 * @brief Mean squared error.
 *
 * The MSE is defined as:
 * \f{eqnarray*}{
 * \mathrm{MSE}(y, w)
 *  &=& \frac{1}{W}\sum_{i=1}^nw_i\left[y_i - \hat\pi(y, w)\right]^2 \\
 *  &=& \frac{1}{W}\left[\sum_{i=1}^nw_iy_i^2 - 2\hat\pi(y, w)\sum_{i=1}^nw_iy_i\right] + \hat\pi(y, w)^2 \\
 *  &=& \frac{1}{W}\left[\mathrm{WSS} - 2W\hat\pi(y, w)^2\right] + \hat\pi(y, w)^2 \\
 *  &=& \frac{1}{W}\mathrm{WSS} - \hat\pi(y, w)^2.
 * \f}
 * where \f$W = \sum_{i=1}^nw_i\f$ is the sum of weights
 * (i.e. the number of samples in the unweighted case),
 * \f$\hat\pi(y, w) = \frac{1}{W}\sum_{i=1}^nw_iy_i\f$ is the weighted prediction of the \f$y_i\f$'s
 * (i.e. the average of the \f$y_i\f$'s in the unweighted case) and
 * \f$\mathrm{WSS} = \sum_{i=1}^nw_iy_i^2\f$ denotes the weighted sum of squares.
 *
 * With this formulation, only \f$\hat\pi(y, w)\f$ and \f$\mathrm{WSS}\f$ need to be
 * maintained at all time for the loss to be evaluated.
 *
 * @tparam FloatType See NodeBasedLoss::FloatType.
 */
DEFINE_NODE_LOSS(MeanSquaredError)
public:
    MeanSquaredError():
            ParentLoss(), weighted_sum_squares{0} {
    }
    MeanSquaredError(const TreeConfig&):
            MeanSquaredError() {
    }
    ~MeanSquaredError() = default;
protected:
    /// \\sum_i w_i y_i²  (WSS)
    Float weighted_sum_squares;

    inline Float compute() const override final {
        Float mu{this->get_mu()};
        return weighted_sum_squares / sum_of_weights - mu*mu;
    }
    inline void _augment(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            weighted_sum += ys[i];
            weighted_sum_squares += ys[i]*ys[i];
        }
        sum_of_weights += ys.size();
    }
    inline void _augment(const Array<Float>& ys,
                         const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            weighted_sum += ys[i]*ws[i];
            weighted_sum_squares += ys[i]*ys[i]*ws[i];
            sum_of_weights += ws[i];
        }
    }
    inline void _augment(const MeanSquaredError<Float>& other_loss) override final {
        weighted_sum += other_loss.weighted_sum;
        weighted_sum_squares += other_loss.weighted_sum_squares;
        sum_of_weights += other_loss.sum_of_weights;
    }
    inline void _diminish(const Array<Float>& ys) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            weighted_sum -= ys[i];
            weighted_sum_squares -= ys[i]*ys[i];
        }
        sum_of_weights -= ys.size();
    }
    inline void _diminish(const Array<Float>& ys,
                          const Array<Float>& ws) override final {
        for(size_t i{0}; i < ys.size(); ++i) {
            weighted_sum -= ys[i]*ws[i];
            weighted_sum_squares -= ys[i]*ys[i]*ws[i];
            sum_of_weights -= ws[i];
        }
    }
    inline void _diminish(const MeanSquaredError<Float>& other_loss) override final {
        weighted_sum -= other_loss.weighted_sum;
        weighted_sum_squares -= other_loss.weighted_sum_squares;
        sum_of_weights -= other_loss.sum_of_weights;
    }
END_OF_DEFINITION

/**
 * @namespace Cart::Loss::impl
 *
 * Local namespace for implementation details that do not need to be used
 * by a user of the library.
 */
namespace impl {
/**
 * @class Cart::Loss::impl::NonNegativeIntegerLoss
 * @brief Generic abstract class for loss when \f$y \in [0, N] \cap \mathbb{Z}\f$.
 *
 * Keeps the sums
 * (i) \f$\displaystyle\sum_{i \in \mathcal{I}_y}w_i\f$ for each integer value of \f$y\f$,
 * (ii) \f$\displaystyle\sum_{i=1}^nw_i\f$,
 * (iii) \f$\displaystyle\sum_{i=1}^ny_i\f$ and
 * (iv) \f$\displaystyle\sum_{i=1}^nw_iy_i\f$.
 *
 * Here, \f$\mathcal{I}_y\f$ is the set of indices \f$i \in [1, n] \cap \mathbb{Z}\f$
 * such that \f$y_i = y\f$.
 *
 * Only compute() remains to be implemented for concrete classes.
 * See PoissonDeviance and NegativeBinomialDeviance.
 */
template <std::floating_point FloatType, class LossType>
class NonNegativeIntegerLoss : public NodeBasedLoss<FloatType, LossType> {
private:
    typedef NodeBasedLoss<FloatType, LossType> ParentLoss;
public:
    using typename ParentLoss::Float;
    NonNegativeIntegerLoss():
            ParentLoss(),
            max_y{0}, sum_wi_when_y(16, 0.), unweighted_sum{0.} {
    }
    NonNegativeIntegerLoss(const TreeConfig&):
            NonNegativeIntegerLoss() {
    }

    virtual ~NonNegativeIntegerLoss() = default;
protected:
    using ParentLoss::sum_of_weights;
    using ParentLoss::weighted_sum;
    size_t max_y;
    std::vector<Float> sum_wi_when_y;
    // \\sum_i y_i
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

/**
 * @brief Poisson deviance.
 *
 * The Poisson deviance is defined as (where \f$K\f$ is the maximum value of the \f$y_i\f$'s):
 * \f{eqnarray*}{
 * d_{\mathcal{P}}(y, w)
 * &=& \frac{2}{W}\left[\sum_{i=1}^nw_i\left(y_i\log\frac{y_i}{\hat\pi(y, w)} + \hat\pi(y, w) - y_i\right)\right] \\
 * &=& \frac{2}{W}\sum_{i=1}^nw_iy_i\log\frac{y_i}{\hat\pi(y, w)} \\
 * &=& \frac{2}{W}\sum_{k=1}^K\sum_{i \in \mathcal{I}_k}w_ik\log\frac{k}{\hat\pi(y, w)} \\
 * &=& \frac{2}{W}\sum_{k=1}^K\left[\left(\sum_{i \in \mathcal{I}_k}w_i\right) \cdot k\log\frac{k}{\hat\pi(y, w)}\right]
 * \f}
 *
 * It is therefore sufficient to maintain \f$W\f$ and \f$\displaystyle\sum_{i \in \mathcal{I}_k}w_i\f$
 * (for every \f$0 \le k \le K\f$) to compute the deviance.
 */
template <std::floating_point FloatType>
class PoissonDeviance final : public impl::NonNegativeIntegerLoss<
                                FloatType,
                                PoissonDeviance<FloatType>
                        > {
private:
    typedef impl::NonNegativeIntegerLoss<
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
        Float mu{this->get_mu()};
        Float ret{0};
        for(size_t y{1}; y <= max_y; ++y)
            ret += sum_wi_when_y[y] * y * std::log(y / mu);
        return 2 * ret / sum_of_weights;
    }
};

/**
 * @brief Negative binomial (of parameter \f$\alpha\f$) deviance.
 *
 * The deviance is defined as:
 * \f{eqnarray*}{
 * d(y, w)
 * &=& \frac{1}{W}\sum_{i=1}^nw_i\left(\frac{1}{\alpha}\log\frac{1+\alpha\hat\pi(y, w)}{1+\alpha y} + y\log\frac{y(1+\alpha\hat\pi(y, w))}{\hat\pi(y, w)(1 + \alpha y)}\right) \\
 * &=& \frac{1}{W}\left(\sum_{i \in \mathcal{I}_0}w_i\right)\log(1+\alpha\hat\pi(y, w))
 *     + \frac{1}{W}\sum_{k=1}^K\left[\left(\sum_{i \in \mathcal{I}_k}w_i\right)\left(\frac{1}{\alpha}\log\frac{1+\alpha\hat\pi(y, w)}{1+\alpha k} + k\log\frac{k(1+\alpha\hat\pi(y, w))}{\hat\pi(y, w)(1+\alpha k)}\right)\right].
 * \f}
 */
template <std::floating_point FloatType>
class NegativeBinomialDeviance final : public impl::NonNegativeIntegerLoss<
                                            FloatType,
                                            NegativeBinomialDeviance<FloatType>
                                > {
private:
    typedef impl::NonNegativeIntegerLoss<
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
        Float mu{this->get_mu()};
        Float ret{sum_wi_when_y[0] * std::log(1 + alpha*mu)};
        for(size_t y{1}; y <= max_y; ++y) {
            ret += sum_wi_when_y[y] * (
                std::log((1 + alpha*mu) / (1 + alpha*y)) / alpha
                + y*std::log((y*(1 + alpha*mu)) / (mu*(1 + alpha*y)))
            );
        }
        return 2 * ret / sum_of_weights;
    }
};

template <std::floating_point Float>
struct SplitPred {
    Float left_size;
    Float left_pred;
    Float right_size;
    Float right_pred;
};

template <std::floating_point FloatType, class LossType>
class TreeBasedLoss {
public:
    typedef FloatType Float;

protected:
    LossType& self;

    const Dataset<Float>& dataset;
    const Node<Float>* current_node{nullptr};
    size_t last_idx{0};
    std::vector<std::pair<Float, Float>> precomputed_modalities;
    Float total_size{0};
    Float total_sum{0};
    Float left_sum{0};
    Float left_sum_of_weights{0};
    Float right_sum{0};
    Float right_sum_of_weights{0};

    virtual void _add_expanded_node(const Node<Float>* node) = 0;
     virtual Float  _evaluate() const = 0;
    // For numerical splits
    virtual Float  _evaluate(
            const Array<Float>& y, size_t idx) const = 0;
    virtual Float  _evaluate(
            const Array<Float>& y, const Array<Float>& w, size_t idx) const = 0;
    // For categorical splits
    virtual Float _evaluate(uint64_t mask) = 0;
    virtual Float _evaluate(uint64_t mask, SplitPred<Float>&) = 0;
    virtual void _set_root(Node<Float>* node) = 0;

    virtual void _new_node() { }
    virtual void _new_feature() { }
public:
    TreeBasedLoss() = delete;

    TreeBasedLoss(const Dataset<Float>& data):
            self{static_cast<LossType&>(*this)}, dataset{data} {
    }

    ~TreeBasedLoss() = default;

    /**
     * @brief Notify the loss that a new node is being looked at for the
     * current split.
     *
     * @param node The node that is considered.
     */
    inline void new_node(const Node<Float>* node) {
        current_node = node;
        self._new_node();
    }

    /**
     * @brief Notify the loss that a new covariate is being looked at for
     * the current split on the current node.
     *
     * @param j The index of the new covariate.
     */
    inline void new_feature(size_t j) {
        auto const& [Xj, y, p, w, indices] = current_node->data->sorted_Xypw(j);
        precomputed_modalities.clear();
        if(current_node->data->is_categorical(j)) {
            total_size = total_sum = 0;
            auto [values, counts] = unique(Xj);
            size_t nb_modalities = counts.size();
            auto sumcounts{cumsum<size_t>(counts)};
            size_t base_idx, idx = 0;
            for(size_t k{0}; k < nb_modalities; ++k) {
                base_idx = idx;
                idx = sumcounts[k];
                auto ys{y.view(base_idx, idx)};
                if(current_node->data->is_weighted()) {
                    auto ws{w.view(base_idx, idx)};
                    precomputed_modalities.emplace_back(
                        sum(ws),
                        weighted_sum(ys, ws)
                    );
                } else {
                    precomputed_modalities.emplace_back(
                        static_cast<Float>(idx - base_idx),
                        sum(ys)
                    );
                }
                total_size += precomputed_modalities.back().first;
                total_sum += precomputed_modalities.back().second;
            }
        } else {
            last_idx = 0;
        }
        left_sum = left_sum_of_weights = 0;
        if(current_node->data->is_weighted()) {
            right_sum = weighted_sum(y, w);
            right_sum_of_weights = sum(w);
        } else {
            right_sum = sum(y);
            right_sum_of_weights = static_cast<Float>(y.size());
        }
        self._new_feature();
    }

    /**
     * @brief Evaluate the loss after splitting the current node according to
     * the value of the
     */
    inline Float evaluate(const Array<Float>& y, size_t idx) const {
        return self._evaluate(y, idx);
    }
    inline Float evaluate(const Array<Float>& y,
                          const Array<Float>& w, size_t idx) const {
        return self._evaluate(y, w, idx);
    }
    inline Float evaluate(uint64_t mask) const {
        return self._evaluate(mask);
    }
    inline Float evaluate(uint64_t mask, SplitPred<Float>& res) const {
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

    /**
     * @brief Notify the loss that a given node has been split.
     */
    inline void add_expanded_node(const Node<Float>* node) {
        assert(not node->is_leaf());
        self._add_expanded_node(node);
    }

    /**
     * @brief Provide the root of the tree.
     *
     * @param node The root of the tree.
     */
    inline void set_root(Node<Float>* node) {
        self._set_root(node);
    }
};

#define __USING_CLAUSE_TREE_LOSS \
protected: \
    using ParentLoss::self; \
    using ParentLoss::dataset; \
    using ParentLoss::current_node; \
    using ParentLoss::precomputed_modalities; \
    using ParentLoss::last_idx; \
    using ParentLoss::total_size; \
    using ParentLoss::total_sum; \
    using ParentLoss::left_sum; \
    using ParentLoss::left_sum_of_weights; \
    using ParentLoss::right_sum; \
    using ParentLoss::right_sum_of_weights; \
public: \
    using typename ParentLoss::Float;

#define DEFINE_TREE_LOSS(NAME) \
template <std::floating_point FloatType> \
class NAME final : public TreeBasedLoss<FloatType, NAME<FloatType>> { \
private: \
    typedef TreeBasedLoss<FloatType, NAME<FloatType>> ParentLoss; \
protected: \
    friend class TreeBasedLoss<FloatType, NAME<FloatType>>; \
    __USING_CLAUSE_TREE_LOSS

template <std::floating_point Float>
using Coord = std::pair<Float, Float>;


template <std::floating_point Float>
struct QuantileFunctionEntry final {
    const Node<Float>* node;
    Float N;
    Float pred;

    QuantileFunctionEntry() = delete;

    QuantileFunctionEntry(const Node<Float>* node, Float N, Float pred):
            node{node}, N{N}, pred{pred} {
    }

    explicit QuantileFunctionEntry(const Node<Float>* node):
            node{node}, N{node->sum_of_weights}, pred{node->pred} {
    }

    inline QuantileFunctionEntry& operator=(const Node<Float>* node) {
        this->node = node;
        this->N = node->sum_of_weights;
        this->pred = node->pred;
        return *this;
    }
};

template <std::floating_point FloatType, bool allow_crossing=true>
class GiniIndexLorenzCurve final : public TreeBasedLoss<
                                    FloatType,
                                    GiniIndexLorenzCurve<FloatType, allow_crossing>
                          > {
    typedef TreeBasedLoss<
        FloatType,
        GiniIndexLorenzCurve<FloatType, allow_crossing>
    > ParentLoss;

    __USING_CLAUSE_TREE_LOSS
public:
    GiniIndexLorenzCurve(const Dataset<Float>& data):
            ParentLoss(data), curve() {
    }

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
        std::vector<QuantileFunctionEntry<Float>> quantiles;
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
                QuantileFunctionEntry<Float> const& entry) {
            auto it{std::find_if(
                first, last,
                [&entry](const QuantileFunctionEntry<Float>& x) -> bool {
                    return x.pred >= entry.pred;
                }
            )};
            std::shift_right(it, last, 1);
            *it = entry;
        }
    };
protected:
    friend class TreeBasedLoss<Float, GiniIndexLorenzCurve<Float, allow_crossing>>;

    size_t nb_modalities{0};

    LorenzCurve curve;

    static inline Float _evaluate(const LorenzCurve& curve) {
        return static_cast<Float>(1) - 2*curve.area();
    }

    inline Float _evaluate() const override final {
        return _evaluate(curve);
    }

    inline void _set_root(Node<Float>* node) override final {
        // Reset the Lorenz curve by calling its constructor.
        // // Call the destructor first!
        curve.~LorenzCurve();
        new(&curve) LorenzCurve(node);
    }

    virtual inline void _add_expanded_node(const Node<Float>* node) override final {
        curve.split_node(node);
    }

    virtual inline Float _evaluate(
            const Array<Float>& y,
            const Array<Float>& w,
            size_t idx) const override final {
        LorenzCurve split_curve(curve);
        auto ws{w.view(last_idx, idx)};
        auto diff{weighted_sum<Float>(y.view(last_idx, idx), ws)};
        auto _this{const_cast<GiniIndexLorenzCurve<Float, allow_crossing>*>(this)};
        _this->last_idx = idx;
        _this->left_sum  += diff;
        _this->right_sum -= diff;
        auto diff_weights{sum(ws)};
        _this->left_sum_of_weights  += diff_weights;
        _this->right_sum_of_weights -= diff_weights;
        return _split_and_evaluate(current_node, split_curve);
    }

    virtual inline Float _evaluate(
            const Array<Float>& y,
            size_t idx) const override final {
        LorenzCurve split_curve(curve);
        auto diff{sum(y.view(last_idx, idx))};
        auto _this{const_cast<GiniIndexLorenzCurve<Float, allow_crossing>*>(this)};
        _this->last_idx = idx;
        _this->left_sum += diff;
        _this->right_sum -= diff;
        _this->left_sum_of_weights  = static_cast<Float>(idx);
        _this->right_sum_of_weights = static_cast<Float>(y.size() - idx);
        return _split_and_evaluate(current_node, split_curve);
    }
    virtual inline Float _evaluate(uint64_t mask) override final {
        return _evaluate(mask, nullptr);
    }

    virtual inline Float _evaluate(
            uint64_t mask, SplitPred<Float>& res) override final {
        return _evaluate(mask, &res);
    }

    Float _evaluate(uint64_t mask, SplitPred<Float>* res) {
        LorenzCurve split_curve(curve);
        left_sum = 0;
        left_sum_of_weights = 0;
        for(size_t mod_idx{0}; mod_idx < nb_modalities; ++mod_idx) {
            if(mask & (1ull << mod_idx)) {
                left_sum += precomputed_modalities[mod_idx].second;
                left_sum_of_weights += precomputed_modalities[mod_idx].first;
            }
        }
        right_sum_of_weights = total_size - left_sum_of_weights;
        right_sum = total_sum - left_sum;
        split_curve.split_node(
            current_node,
            left_sum_of_weights, left_sum / left_sum_of_weights,
            right_sum_of_weights, right_sum / right_sum_of_weights
        );
        if(res != nullptr) [[likely]] {
            res->left_size = left_sum_of_weights;
            res->left_pred = left_sum / left_sum_of_weights;
            res->right_size = right_sum_of_weights;
            res->right_pred = right_sum / right_sum_of_weights;
        }
        return _evaluate(split_curve);
    }

private:
    inline Float _split_and_evaluate(const Node<Float>* node, LorenzCurve& curve) const {
        curve.split_node(
            node,
            left_sum_of_weights, left_sum / left_sum_of_weights,
            right_sum_of_weights, right_sum / right_sum_of_weights
        );
        if constexpr(not allow_crossing) {
            if(curve.crosses(curve))
                return -std::numeric_limits<Float>::infinity();
        }
        return _evaluate(curve);
    }
};

template <std::floating_point Float>
using NonCrossingLorenzCurveError = GiniIndexLorenzCurve<Float, false>;
template <std::floating_point Float>
using CrossingLorenzCurveError = GiniIndexLorenzCurve<Float, true>;

template <std::floating_point Float>
static inline auto _consecutive_lcs(const std::vector<Node<Float>*>& nodes) {
    typename GiniIndexLorenzCurve<Float>::LorenzCurve lc(nodes.front());
    std::vector<typename Cart::Loss::GiniIndexLorenzCurve<Float>::LorenzCurve> ret;
    ret.push_back(lc);
    for(const Node<Float>* node : nodes) {
        lc.split_node(node);
        ret.push_back(lc);
    }
    return ret;
}

DEFINE_TREE_LOSS(GiniIndexABL)
public:
    GiniIndexABL(const Dataset<Float>& data):
            ParentLoss(data), tot_weight(data.weighted_size()) {
    }

protected:
    virtual void _set_root(Node<Float>* node) override final {
        nodes.emplace_back(node);
    }

    virtual void _new_node() override final {
        assert(current_node != nullptr);
        base_value = get_value() - get_contribution_of(current_node);
    }

    virtual void _add_expanded_node(const Node<Float>* node) override final {
        auto& entry{get_entry(node)};
        entry = node->left_child;
        nodes.emplace_back(node->right_child);
        precomputed = false;
    }

    inline Float get_value() const {
        if(not precomputed)  [[unlikely]] {
            Float value{0};
            for(auto it1{nodes.begin()}; it1 != nodes.end(); ++it1)
                for(auto it2{nodes.begin()}; it2 != it1; ++it2)
                    value += it1->N * it2->N * std::fabs(it1->pred - it2->pred);
            value *= 2 / (tot_weight * tot_weight);
            const_cast<GiniIndexABL*>(this)->precomputed = true;
            const_cast<GiniIndexABL*>(this)->value = value;
        }
        return value;
    }

    virtual inline Float _evaluate() const override final {
        //return static_cast<Float>(1) - 2*get_value();
        return get_value();
    }

    virtual inline Float _evaluate(
            const Array<Float>& y,
            size_t idx) const override final {
        auto diff{sum(y.view(last_idx, idx))};
        auto _this{const_cast<GiniIndexABL*>(this)};
        _this->last_idx = idx;
        _this->left_sum += diff;
        _this->right_sum -= diff;
        _this->left_sum_of_weights  = static_cast<Float>(idx);
        _this->right_sum_of_weights = static_cast<Float>(y.size() - idx);
        return __evaluate();
    }

    virtual inline Float _evaluate(
            const Array<Float>& y,
            const Array<Float>& w,
            size_t idx) const override final {
        auto ws{w.view(last_idx, idx)};
        auto diff{weighted_sum(y.view(last_idx, idx), ws)};
        auto _this{const_cast<GiniIndexABL*>(this)};
        _this->last_idx = idx;
        _this->left_sum += diff;
        _this->right_sum -= diff;
        auto diff_weights{sum(ws)};
        _this->left_sum_of_weights  += diff_weights;
        _this->right_sum_of_weights -= diff_weights;
        return __evaluate();
    }

    virtual inline Float _evaluate(uint64_t mask) override final {
        return _evaluate(mask, nullptr);
    }

    virtual inline Float _evaluate(
            uint64_t mask, SplitPred<Float>& res) override final {
        return _evaluate(mask, &res);
    }

    Float _evaluate(uint64_t mask, SplitPred<Float>* res) {
        const auto nb_modalities{precomputed_modalities.size()};
        left_sum = 0;
        left_sum_of_weights = 0;
        for(size_t mod_idx{0}; mod_idx < nb_modalities; ++mod_idx) {
            if(mask & (1ull << mod_idx)) {
                left_sum += precomputed_modalities[mod_idx].second;
                left_sum_of_weights += precomputed_modalities[mod_idx].first;
            }
        }
        right_sum_of_weights = total_size - left_sum_of_weights;
        right_sum = total_sum - left_sum;
        if(res != nullptr) [[likely]] {
            res->left_size = left_sum_of_weights;
            res->left_pred = left_sum / left_sum_of_weights;
            res->right_size = right_sum_of_weights;
            res->right_pred = right_sum / right_sum_of_weights;
        }
        return __evaluate();
    }

    bool precomputed{false};
    Float value{0};
    Float base_value{0};
    Float tot_weight{0};

    std::vector<QuantileFunctionEntry<Float>> nodes;

private:
    inline Float __evaluate() const {
        auto pred_left{left_sum / left_sum_of_weights};
        auto pred_right{right_sum / right_sum_of_weights};
        auto value{base_value
            + get_contribution_of(current_node, left_sum_of_weights, pred_left)
            + get_contribution_of(current_node, right_sum_of_weights, pred_right)
            + 2 * left_sum_of_weights * right_sum_of_weights
                * std::fabs(pred_left - pred_right)
                / (tot_weight * tot_weight)};
        return value;
        // return 1 - 2*value;
    }

    inline Float get_contribution_of(const Node<Float>* node) const {
        Float ret{0};
        auto const& entry{get_entry(node)};
        auto prop_leaf{entry.N};
        auto pred_leaf{entry.pred};
        for(const auto& entry : nodes)
            if(entry.node != current_node)  [[likely]]
                ret += entry.N * std::fabs(pred_leaf - entry.pred);
        return 2*prop_leaf*ret / (tot_weight*tot_weight);
    }

    inline Float get_contribution_of(
            const Node<Float>* node,
            Float prop, Float pred) const {
        Float ret{0};
        for(const auto& entry : nodes)
            if(entry.node != node)  [[likely]]
            ret += entry.N * std::fabs(pred - entry.pred);
        return 2*prop*ret / (tot_weight*tot_weight);
    }

    inline QuantileFunctionEntry<Float>& get_entry(const Node<Float>* node) {
        auto it{std::find_if(
            nodes.begin(), nodes.end(), [node](const auto& n) -> bool {
                return n.node == node;
            }
        )};
        assert(it != nodes.end());
        return *it;
    }

    inline const QuantileFunctionEntry<Float>& get_entry(
            const Node<Float>* node) const {
        return const_cast<GiniIndexABL*>(this)->get_entry(node);
    }
END_OF_DEFINITION

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

// template <Loss::_TreeBasedLoss LossType>
// struct CanBeDepthFirst<LossType> {
//     static constexpr bool value{false};
// };

template <Loss::_Loss LossType>
struct CanBeBestFirst {
    static constexpr bool value{true};
};


}  // Cart::Loss::
}  // Cart::

#undef DEFINE_NODE_LOSS
#undef DEFINE_TREE_LOSS
#undef END_OF_DEFINITION

#endif
