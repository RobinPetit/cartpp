#ifndef CART_SORT_HPP
#define CART_SORT_HPP

#include "array.hpp"

namespace Cart {
namespace Sort {

/**
 * @brief Sorting algorithm.
 */
enum class SortingAlgorithm : int {
    /// Merge sort
    MERGESORT,
    /// 3-way quicksort (partition with `[< | = | >]` instead of `[<= | >]`)
    QUICKSORT_3WAY
};


/********** Merge Sort **********/

namespace {
template <typename Comp, typename T>
static void merge(
            T* data, T* aux, size_t beg, size_t mid, size_t end,
            const Comp& comp) {
    std::memcpy(aux+beg, data+beg, (end-beg) * sizeof(T));
    size_t i{beg};
    size_t j{mid};
    for(size_t k{beg}; k < end; ++k) {
        if(i >= mid or (j < end and comp(aux[j], aux[i]))) {
            data[k] = aux[j++];
        } else {
            data[k] = aux[i++];
        }
    }
}
}

/**
 * @brief Implementation of mergesort on Array.
 *
 * Should not be called directly, go through Cart::Sort::sort() instead.
 *
 * @tparam T Type of array's items.
 * @tparam Comp Type of array's items comparator.
 *   Defaults to \link std::less std::less<T>\endlink
 */
template <typename Comp, typename T>
static void mergesort(Array<T>& array, const Comp& comp) {
    size_t N{array.size()};
    T* aux{new T[N]};
    for(size_t size{1}; size < N; size <<= 1) {
        for(size_t idx{0}; idx+size < N; idx += 2*size) {
            merge<Comp, T>(
                std::addressof(array[0]), aux,
                idx, idx+size, std::min(idx+2*size, N),
                comp
            );
        }
    }
    delete[] aux;
    aux = nullptr;
}

/********** Quicksort **********/

namespace {
template <typename Comp, typename T>
static inline std::pair<size_t, size_t> partition_3way(
        T* data, size_t beg, size_t end, const Comp& comp) {
    size_t i{beg};
    size_t j{beg+1};
    size_t k{end};
    auto pivot{data[beg]};  // TODO: choose at random?
    while(j < k) {
        if(comp(data[j], pivot)) {
            std::swap(data[j++], data[i++]);
        } else if(comp(pivot, data[j])) {
            std::swap(data[--k], data[j]);
        } else {
            ++j;
        }
    }
    return {i, k};
}

template <typename Comp, typename T>
static inline void quicksort_3way(
        T* data, size_t beg, size_t end, const Comp& comp) {
    if(end <= beg) [[unlikely]]
        return;
    auto [pivot_left, pivot_right] = partition_3way(data, beg, end, comp);
    quicksort_3way(data, beg, pivot_left, comp);
    quicksort_3way(data, pivot_right, end, comp);
}
}

/**
 * @brief Implementation of 3-way quicksort on Array.
 *
 * Should not be called directly, go through Cart::Sort::sort() instead.
 *
 * @tparam T Type of array's items.
 * @tparam Comp Type of array's items comparator.
 *   Defaults to \link std::less std::less<T>\endlink
 */
template <typename Comp, typename T>
static inline void quicksort_3way(Array<T>& array, const Comp& comp) {
    quicksort_3way(std::addressof(array[0]), 0, array.size(), comp);
}

/**
 * @brief Sort an array inplace.
 *
 * @tparam T Type of array's items.
 * @tparam Comp Type of array's items comparator.
 *   Defaults to \link std::less std::less<T>\endlink.
 * @param array The array to sort.
 * @param algo Algorithm used to sort.
 *   Defaults to mergesort.
 * @param comp Instance of comparator.
 *   Usually not needed.
 */
template <typename T, typename Comp=std::less<T>>
static inline void sort(Array<T>& array,
          SortingAlgorithm algo=SortingAlgorithm::MERGESORT,
          Comp comp=Comp()) {
    array.ensure_contiguous();
    switch(algo) {
    case SortingAlgorithm::MERGESORT:
        mergesort(array, comp);
        break;
    case SortingAlgorithm::QUICKSORT_3WAY:
        quicksort_3way(array, comp);
        break;
    }
}

/**
 * @brief Get a sorted copy of an array.
 *
 * @tparam T Type of array's items.
 * @tparam Comp Type of array's items comparator.
 *   Defaults to \link std::less std::less<T>\endlink.
 * @param array The array to sort.
 * @param algo The algorithm to sort with.
 *   Defaults to mergesort.
 * @param comp Instance of comparator.
 *   Usually not needed.
 */
template <typename T, typename Comp=std::less<T>>
static inline Array<T> sorted(const Array<T>& array,
                       SortingAlgorithm algo=SortingAlgorithm::MERGESORT,
                       Comp comp=Comp()) {
    auto copy(array.copy());
    sort(copy, algo, comp);
    return copy;
}

namespace {
template <typename T>
struct _ArgsortKey {
    using ValueType = T;
    const T* base_ptr;
    _ArgsortKey(const Array<T>& a):
            base_ptr{&(*a.begin())} {
    }
    inline bool operator()(size_t i, size_t j) const {
        return base_ptr[i] < base_ptr[j];
    }
};
}

/**
 * @brief Get a permutation of the indices that sorts a given array.
 *
 * @tparam T Type of array's items.
 * @param array The array so sort.
 * @param method The algorithm to sort with.
 */
template <typename T>
static inline Array<size_t> argsort(const Array<T>& array, SortingAlgorithm method) {
    Array<size_t> indices{range(0, array.size())};
    _ArgsortKey<T> key(array);
    sort(indices, method, key);
    // assert(std::is_sorted(
    //     indices.begin(), indices.end(),
    //     [&array](size_t i, size_t j) -> bool {
    //         return array[i] < array[j];
    //     }
    // ));
    return indices;
}

}  // Cart::Sort::

namespace {
template <typename T>
static inline std::pair<std::vector<T>, std::vector<size_t>> unique_of_sorted(
        const Array<T>& sorted_array) {
    if(sorted_array.size() == 0) [[unlikely]]
        return {};
    std::vector<size_t> counts(1, 1);
    std::vector<T> values(1, sorted_array[0]);
    for(size_t i{1}; i < sorted_array.size(); ++i) {
        if(sorted_array[i] == sorted_array[i-1]) {
            ++counts.back();
        } else {
            counts.push_back(1);
            values.push_back(sorted_array[i]);
        }
    }
    assert(values.size() == counts.size());
    return {values, counts};
}
}

/**
 * @brief Get the unique values from array and the number of times it appeared.
 *
 * @tparam T The type of array's items.
 * @param array The array to process.
 * @param is_sorted `true` if array is already sorted, `false` otherwise.
 */
template <typename T>
static inline std::pair<std::vector<T>, std::vector<size_t>> unique(
        const Array<T>& array, bool is_sorted=true) {
    if(is_sorted) {
        return unique_of_sorted(array);
    } else {
        auto _sorted_array{Sort::sorted(array)};
        return unique_of_sorted(_sorted_array);
    }
}

}  // Cart::

#endif
