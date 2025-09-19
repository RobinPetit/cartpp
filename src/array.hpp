#ifndef CART_ARRAY_HPP
#define CART_ARRAY_HPP

#include <cassert>
#include <concepts>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "utils.hpp"

namespace Cart {

template <typename T>
class Array;

inline Array<size_t> where(const Array<bool>&);

template <typename T>
class Array {
private:
    template <typename VecType>
    struct GenericArrayIterator {
    public:
        using value_type = T;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        GenericArrayIterator(VecType* vec):
            vector{vec}, i{0} {
        }
        inline VecType& operator*() {
            return vector[i];
        }
        // Post-decrement
        inline GenericArrayIterator operator--(int) {
            GenericArrayIterator ret(*this);
            --*this;
            return ret;
        }
        // Pre-decrement
        inline GenericArrayIterator& operator--() {
            --i;
            return *this;
        }

        inline GenericArrayIterator& operator-=(size_t offset) {
            i -= offset;
            return *this;
        }

        inline GenericArrayIterator operator-(size_t offset) const {
            GenericArrayIterator ret(*this);
            return ret -= offset;
        }
        // Post-increment
        inline GenericArrayIterator operator++(int) {
            GenericArrayIterator ret(*this);
            ++*this;
            return ret;
        }
        // Pre-increment
        inline GenericArrayIterator& operator++() {
            ++i;
            return *this;
        }

        inline GenericArrayIterator& operator+=(size_t offset) {
            i += offset;
            return *this;
        }

        inline GenericArrayIterator operator+(size_t offset) const {
            GenericArrayIterator ret(*this);
            return ret += offset;
        }

        inline difference_type operator-(const GenericArrayIterator& other) const {
            difference_type ret = (vector+i) - (other.vector+other.i);
            return ret;
        }

        inline bool operator==(const GenericArrayIterator& other) const {
            return  vector == other.vector
                and i == other.i;
        }
    private:
        VecType* vector;
        ssize_t i;
    };
private:
    explicit Array(std::pair<T*, size_t> args):
            data{args.first}, n{args.second}, owns_data{true} {
    }
public:
    typedef GenericArrayIterator<      T>      ArrayIterator;
    typedef GenericArrayIterator<const T> ConstArrayIterator;
    Array(): Array(0) {}
    Array(size_t size):
            data{new T[size]}, n{size}, owns_data{true} {
    }
    Array(size_t size, T value):
            Array(size) {
        for(size_t i{0}; i < n; ++i)
            data[i] = value;
    }
    Array(T* ptr, size_t size, bool copy=true):
            data{copy ? new T[size] : ptr}, n{size}, owns_data{copy} {
        if(copy)
            std::memcpy(data, ptr, sizeof(T) * n);
    }
    explicit Array(const Array<T>& other):
            data{other.data}, n{other.n}, owns_data{false} {
    }
    Array(Array<T>&& other):
            data{std::move(other.data)}, n{other.n}, owns_data{other.owns_data} {
        other.data = nullptr;
        other.n = 0;
        other.owns_data = false;
    }
    Array(const std::vector<T>& vector):
            data{new T[vector.size()]}, n{vector.size()}, owns_data{true} {
        for(size_t i{0}; i < size(); ++i)
            data[i] = vector[i];
    }
    ~Array() {
        if(owns_data and data != nullptr)
            delete[] data;
        data = nullptr;
        owns_data = false;
    }
    Array& operator=(const Array<T>&) = delete;
    Array& operator=(Array<T>&& other) {
        assert(data != other.data);
        if(owns_data and data != nullptr)
            delete[] data;
        data = other.data;
        n = other.n;
        owns_data = other.owns_data;
        other.data = nullptr;
        other.owns_data = false;
        return *this;
    }
    inline Array<std::remove_const_t<T>> copy() const {
        Array<std::remove_const_t<T>> ret(n);
        fill_copy(ret);
        return ret;
    }

    enum class BinaryFormat {
        NATIVE
    };

    void save_to(std::ostream& out, BinaryFormat fmt=BinaryFormat::NATIVE) const {
        switch(fmt) {
        case BinaryFormat::NATIVE:
            out.write(AS_CONSTCHARPTR(&n), ssizeof(n));
            out.write(AS_CONSTCHARPTR(data), ssizeof(T)*n);
            break;
        default:
            throw std::runtime_error("Unknown format");
        };
    }

    static Array<T> load_from(
            std::istream& in, BinaryFormat fmt=BinaryFormat::NATIVE) {
        size_t n;
        char* buffer{nullptr};
        switch(fmt) {
        case BinaryFormat::NATIVE:
            in.read(AS_CHARPTR(&n), ssizeof(n));
            buffer = new char[n * sizeof(T)];
            in.read(buffer, ssizeof(T)*static_cast<ssize_t>(n));
            break;
        default:
            throw std::runtime_error("Unknown format");
        }
        assert(buffer != nullptr);
        // DO NOT delete[] buffer since ownership was given to ret
        // Use of special private constructor giving ownership of the pointer.
        Array<T> ret(
            std::make_pair(
                static_cast<T*>(static_cast<void*>(buffer)),
                n
            )
        );
        return ret;
    }

    inline Array<T> view(size_t beg, size_t end) const {
        return Array<T>(data+beg, end-beg, false);
    }

    inline ArrayIterator begin() {
        return data;
    }
    inline ArrayIterator end() {
        return iterator_at(n);
    }
    inline ConstArrayIterator begin() const {
        return cbegin();
    }
    inline ConstArrayIterator end() const {
        return cend();
    }
    inline ConstArrayIterator cbegin() const {
        return data;
    }
    inline ConstArrayIterator cend() const {
        return const_iterator_at(n);
    }

    inline const T& operator[](size_t i) const {
        return *const_iterator_at(i);
    }
    inline T& operator[](size_t i) {
        return *iterator_at(i);
    }

    Array<T> operator[](const Array<size_t>& indices) const {
        Array<T> ret(indices.size());
        for(size_t i{0}; i < indices.size(); ++i) {
            ret[i] = (*this)[indices[i]];
        }
        return ret;
    }

    inline Array<T> operator[](const Array<bool>& mask) const {
        return (*this)[where(mask)];
    }

    inline void assign(const Array<T>& other) {
        if(size() != other.size())
            throw std::runtime_error("Size mismatch");
        std::memcpy(data, other.data, size()*sizeof(T));
    }
    inline void assign(T value) {
        for(size_t i{0}; i < size(); ++i)
            data[i] = value;
    }

    inline void ensure_contiguous() const {
        ;  // nop
    }
    inline size_t size() const {
        return n;
    }

    template <typename Comp=std::less<T>>
    void sort(SortingAlgorithm algo=SortingAlgorithm::MERGESORT,
              Comp comp=Comp()) {
        ensure_contiguous();
        switch(algo) {
        case SortingAlgorithm::MERGESORT:
            mergesort(comp);
            break;
        case SortingAlgorithm::QUICKSORT_3WAY:
            quicksort_3way(comp);
            break;
        }
    }

protected:
    T* data;
    size_t n;
    bool owns_data;

    void fill_copy(Array<T>& copy) const {
        assert(copy.size() == this->size());
        copy.ensure_contiguous();  // Should always be ok, but just to make sure
        for(size_t i{0}; i < size(); ++i)
            copy.data[i] = (*this)[i];
    }

    inline ArrayIterator iterator_at(size_t pos) {
        ArrayIterator ret{begin()};
        ret += pos;
        return ret;
    }
    inline ConstArrayIterator const_iterator_at(size_t pos) const {
        ConstArrayIterator ret{begin()};
        ret += pos;
        return ret;
    }

    template <typename Comp>
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
    template <typename Comp>
    static inline void quicksort_3way(
            T* data, size_t beg, size_t end, const Comp& comp) {
        if(end <= beg) [[unlikely]]
            return;
        auto [pivot_left, pivot_right] = partition_3way(data, beg, end, comp);
        quicksort_3way(data, beg, pivot_left, comp);
        quicksort_3way(data, pivot_right, end, comp);
    }
    template <typename Comp>
    inline void quicksort_3way(const Comp& comp) {
        quicksort_3way(data, 0, size(), comp);
    }

    template <typename Comp>
    void mergesort(const Comp& comp) {
        size_t N{size()};
        T* aux{new T[N]};
        for(size_t size{1}; size < N; size <<= 1) {
            for(size_t idx{0}; idx+size < N; idx += 2*size) {
                Array<T>::merge<Comp>(
                    data, aux,
                    idx, idx+size, std::min(idx+2*size, N),
                    comp
                );
            }
        }
        delete[] aux;
    }

    template <typename Comp>
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
};

/********** Utils **********/

inline Array<size_t> where(const Array<bool>& mask) {
    size_t nb_true{0};
    for(size_t i{0}; i < mask.size(); ++i)
        if(mask[i])
            ++nb_true;
    Array<size_t> ret(nb_true);
    size_t j{0};
    for(size_t i{0}; i < mask.size(); ++i) {
        if(mask[i]) {
            ret[j] = i;
            ++j;
        }
    }
    return ret;
}

inline Array<size_t> range(size_t beg, size_t end) {
    Array<size_t> ret(end-beg);
    for(size_t i{0}; i < ret.size(); ++i)
        ret[i] = beg+i;
    return ret;
}

template <typename T>
inline Array<std::remove_const_t<T>> sorted(const Array<T>& array) {
    auto copy(array.copy());
    copy.sort();
    return copy;
}

namespace impl {
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

template <typename T>
inline Array<size_t> argsort(const Array<T>& array, SortingAlgorithm method) {
    Array<size_t> indices{range(0, array.size())};
    impl::_ArgsortKey<T> key(array);
    indices.sort(method, key);
    // assert(std::is_sorted(
    //     indices.begin(), indices.end(),
    //     [&array](size_t i, size_t j) -> bool {
    //         return array[i] < array[j];
    //     }
    // ));
    return indices;
}

template <typename T, typename U>
requires(std::is_convertible_v<U, T>)
inline T _typed_sum(const Array<U>& array) {
    T ret{0};
    for(size_t i{0}; i < array.size(); ++i)
        ret += array[i];
    return ret;
}

template <typename T>
inline T sum(const Array<T>& array) {
    return _typed_sum<T, T>(array);
}

template <typename T, std::floating_point Float=double>
inline Float mean(const Array<T>& array) {
    return _typed_sum<Float, T>(array) / static_cast<Float>(array.size());
}

template <std::floating_point Float>
inline Float weighted_mean(
        const Array<Float>& array, const Array<Float>& weights) {
    Float sum{0};
    // TODO: ensure array.size() == weights.size()
    for(size_t i{0}; i < array.size(); ++i)
        sum += array[i]*weights[i];
    return sum / static_cast<Float>(array.size());
}

template <std::floating_point Float, bool value>
static inline Float weighted_prop_eq(
        const Array<bool>& array, const Array<Float>& weights) {
    // TODO: Ensure array.size() == weights.size()
    Float num{0};
    Float den{0};
    for(size_t i{0}; i < array.size(); ++i) {
        if(array[i] == value)
            num += weights[i];
        den += weights[i];
    }
    return num / den;
}

template <std::floating_point Float>
static inline Float weighted_prop_true(
        const Array<bool>& array, const Array<Float>& weights) {
    return weighted_prop_eq<Float, true>(array, weights);
}
template <std::floating_point Float>
static inline Float weighted_prop_false(
        const Array<bool>& array, const Array<Float>& weights) {
    return weighted_prop_eq<Float, false>(array, weights);
}

template <typename T>
static inline size_t nb_unique(const Array<T>& sorted_array) {
    size_t ret{1};
    for(size_t i{1}; i < sorted_array.size(); ++i)
        if(sorted_array[i] != sorted_array[i-1])
            ++ret;
    return ret;
}

template <typename T>
static inline std::pair<std::vector<T>, std::vector<size_t>> unique(
        const Array<T>& sorted_array, bool is_sorted=true) {
    if(not is_sorted) {
        auto _sorted_array{sorted(sorted_array)};
        return unique(_sorted_array, true);
    }
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

template <typename T>
static inline Array<T> cumsum(const Array<T>& array) {
    Array<T> ret(array.size(), 0);
    ret[0] = array[0];
    for(size_t i{1}; i < array.size(); ++i)
        ret[i] = ret[i-1] + array[i];
    return ret;
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Array<T>& array) {
    for(auto const& x : array)
        os << x;
    return os;
}


}  // Cart::

#endif
