#ifndef CART_CONCEPTS_HPP
#define CART_CONCEPTS_HPP

#include "array.hpp"

namespace Cart {

template <typename Mapper>
concept SortKey = requires(
    const Mapper& m, const Mapper::ValueType* ptr, size_t i, size_t j) {
    {m(ptr, i, j)} -> std::same_as<bool>;
};

template <std::floating_point Float, bool>
struct SplitResult;

template <std::floating_point Float>
struct SplitResult<Float, true> {
    typedef std::pair<Array<Float>, Array<Float>> type;
};

template <std::floating_point Float>
struct SplitResult<Float, false> {
    typedef Array<Float> type;
};

template <typename T, typename Float, bool weighted>
concept NumericalSplitCallback = requires(
        Float x,
        T t,
        typename SplitResult<Float, weighted>::type params) {
    { t(x, params) } -> std::same_as<void>;
};
}  // Cart::

#endif  // CART_CONCEPTS_HPP
