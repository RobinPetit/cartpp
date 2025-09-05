#ifndef CART_UTILS_HPP
#define CART_UTILS_HPP

#include <cstddef>
#include <concepts>

#ifndef _POSIX_C_SOURCE  // Not POSIX-compliant
#include <iostream>
typedef std::streamsize ssize_t;
#endif

namespace Cart {

#define ssizeof(x) static_cast<ssize_t>(sizeof(x))

template <typename Mapper>
concept SortKey = requires(
    const Mapper& m, const Mapper::ValueType* ptr, size_t i, size_t j
) {
    {m(ptr, i, j)} -> std::same_as<bool>;
};

// TODO: add quicksort (especially 3-way partitioning) and compare computation time
enum class SortingAlgorithm : int {
    MERGESORT
};

template <typename T>
static inline char* AS_CHARPTR(T* ptr) {
    return static_cast<char*>(static_cast<void*>((ptr)));
}
template <typename T>
static inline const char* AS_CONSTCHARPTR(T* ptr) {
    return static_cast<const char*>(static_cast<const void*>((ptr)));
}
}  // Cart::

#endif
