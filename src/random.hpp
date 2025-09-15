#ifndef CART_RANDOM_HPP
#define CART_RANDOM_HPP

#include <random>

#include "array.hpp"

namespace Cart {
namespace Random {

template <typename T>
static inline void permutation(Array<T>& array) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::shuffle(array.begin(), array.end(), rng);
}

static inline Array<size_t> permutation(size_t n) {
    Array<size_t> ret{range(0, n)};
    permutation(ret);
    return ret;
}

static inline Array<size_t> choice(size_t n, size_t k, bool replace) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> gen(0, n-1);
    Array<size_t> ret(k);
    if(replace) {
        for(size_t j{0}; j < k; ++j)
            ret[j] = gen(rng);
    } else {
        if(n < k) [[unlikely]]
            throw std::runtime_error("Unable to choose " + std::to_string(k) + " from " + std::to_string(n));
        Array<size_t> tmp = permutation(n);
        ret.assign(tmp.view(0, k));
    }
    return ret;
}

template <typename T>
static inline Array<T> choice(const Array<T>& array, size_t k, bool replace) {
    return array[choice(array.size(), k, replace)];
}

}  // Cart::Random
}  // Cart::

#endif  // CART_RANDOM_HPP
