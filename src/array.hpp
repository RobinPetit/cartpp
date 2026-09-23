#ifndef CART_ARRAY_HPP
#define CART_ARRAY_HPP

#include <cassert>
#include <concepts>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "utils.hpp"

namespace Cart {

template <typename T>
class Array;

template <typename T>
static inline std::ostream& operator<<(std::ostream&, const Array<T>&);

inline Array<size_t> where(const Array<bool>&);

/**
 * @brief Generic one-dimensional array.
 *
 * An Array is contiguous in memory.
 * It is iterable, and allows for random access.
 * An Array can be owner of its data or can be simply a memory view
 * on an existing pointer.
 *
 * Arrays are fixed-size and cannot be resized (hence not like std::vector).
 */
template <typename T>
class Array {
private:
    template <typename VecType>
    /**
     * @bief Iterator for Array.
     */
    struct GenericArrayIterator {
    public:
        using value_type = VecType;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        GenericArrayIterator(pointer vec):
            vector{vec}, i{0} {
        }

        inline reference operator*() {
            return vector[i];
        }

        inline std::add_const_t<value_type>& operator*() const {
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
        pointer vector;
        ssize_t i;
    };
private:
    /**
     * @brief Custom internal constructor. Will never be called outside.
     */
    explicit Array(std::pair<T*, size_t> args):
            data{args.first}, n{args.second}, owns_data{true} {
    }
public:
    /// Iterator for Array.
    typedef GenericArrayIterator<      T>      ArrayIterator;
    /// Const iterator for Array.
    typedef GenericArrayIterator<const T> ConstArrayIterator;

    /**
     * @brief Default constructor
     *
     * No memory is allocated, the Array contains simply nullptr.
     */
    Array():
            data{nullptr}, n{0}, owns_data{false} {
    }

    /**
     * @brief Construct an array of given size, items are constructed by default.
     *
     * @brief size Size of the array to create.
     */
    Array(size_t size):
            data{new T[size]}, n{size}, owns_data{true} {
    }

    /**
     * @brief Construct an array of given size, items are constructed by copy of value.
     *
     * @param size Size of the array to create.
     * @param value The value to copy to every item.
     */
    Array(size_t size, T value):
            Array(size) {
        for(size_t i{0}; i < n; ++i)
            data[i] = value;
    }

    /**
     * @brief Construct an array of given size from the provided pointer.
     *
     * The constructed array is either a copy of the received array
     * or a memory view on the provided array.
     *
     * @param ptr The reference array.
     * @param size The size of the reference array.
     * @param copy Whether or not the constructed array must copy elements of ptr.
     *   If `true`, then a new array is created and all items are copied.
     *   If `false`, then only the pointer is copied and the elements are shared.
     */
    Array(T* ptr, size_t size, bool copy=true):
            data{copy ? new T[size] : ptr}, n{size}, owns_data{copy} {
        if(copy)
            fill_copy(ptr, data, size);
    }

    /**
     * @brief Copy constructor.
     *
     * @param other The array to copy.
     */
    explicit Array(const Array& other):
            data{other.data}, n{other.n}, owns_data{false} {
    }

    /**
     * @brief Move constructor.
     *
     * @param other The array to move.
     */
    Array(Array&& other):
            data{std::move(other.data)}, n{other.n}, owns_data{other.owns_data} {
        other.data = nullptr;
        other.n = 0;
        other.owns_data = false;
    }

    /**
     * @brief Construct an array by copying all elements from a `vector`.
     *
     * @param vector The vector whose elements are copy-constructed from.
     */
    Array(const std::vector<T>& vector):
            data{
                static_cast<T*>(::operator new(sizeof(T) * vector.size()))
            }, n{vector.size()}, owns_data{true} {
        for(size_t i{0}; i < size(); ++i)
            std::construct_at(data+i, vector[i]);
    }

    /**
     * @brief Destructor of `Array`.
     *
     * The memory is freed iff the instance has ownership of its data.
     * In particular, no memory is freed if the object is constructor
     * with `Array(ptr, size, copy=false)`.
     */
    ~Array() {
        if(owns_data and data != nullptr)
            delete[] data;
        data = nullptr;
        owns_data = false;
    }

    Array& operator=(const Array&) = delete;

    /**
     * @brief Move operator.
     *
     * @param other The array to move
     */
    Array& operator=(Array&& other) {
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

    /**
     * @brief Copy an array from `this`.
     */
    inline Array<std::remove_const_t<T>> copy() const {
        Array<std::remove_const_t<T>> ret(*this);
        return ret;
    }

    /**
     * @brief Format for binary encoding in a file.
     */
    enum class BinaryFormat {
        /// Native encoding on the OS
        NATIVE,
        /// Big endian (UNUSED)
        BIG,
        /// Little endian (UNUSED)
        LITTLE,
    };

    /**
     * @brief Write the array on a binary stream (typically a file).
     *
     * Undefined behaviour if T is not a primitive type or a pointer type.
     *
     * TODO: Define properly a format `cartbin` that includes:
     * 1. version
     * 2. binary format (big/little endian)
     * 3. the type T
     *
     * @param out The stream to write to.
     * @param fmt The binary format to use when writing elements of the array.
     */
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

    /**
     * @brief Load an array from a binary stream (typically a file).
     *
     * See Array::save_to.
     * `fmt` should match the format sued when Array::save_to was called.
     * If `fmt` is BinaryFormat::NATIVE, then saving from one system and loading
     * from another is undefined behaviour.
     */
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

    /**
     * @brief Create a view from the instance.
     *
     * The returned array shares the same memory as the instance,
     * no copy is performed.
     *
     * WARNING: No check is performed on the provided indices!
     *
     * @param beg The index to start the view on.
     * @param end The first index to not be part of the view.
     */
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

    /**
     * @brief Get a const reference to the object at index `i`.
     *
     * WARNING: no check is performed on the index!
     *
     * @param i The index to read from.
     */
    inline const T& operator[](size_t i) const {
        return *const_iterator_at(i);
    }

    /**
     * @brief Get a reference to the object at index `i`.
     *
     * WARNING: no check is performed on the index!
     *
     * @param i The index to read from.
     */
    inline T& operator[](size_t i) {
        return *iterator_at(i);
    }

    /**
     * @anchor Array_at_indices
     *
     * @brief Create a new array with a selection of indices (possible duplicates).
     *
     * The received array satisfies `(array[indices])[i] == array[indices[i]]`.
     * All elements are copy-constructed.
     *
     * @param indices The array of indices to select.
     */
    Array<T> operator[](const Array<size_t>& indices) const {
        T* ptr{static_cast<T*>(::operator new(sizeof(T) * indices.size()))};
        for(size_t i{0}; i < indices.size(); ++i)
            std::construct_at(ptr+i, (*this)[indices[i]]);
        Array<T> ret (std::make_pair(ptr, indices.size()));
        return ret;
    }

    /**
     * @anchor Array_at_mask
     *
     * @brief Create a new array by a mask.
     *
     * Create a new array `arr` such that `arr[i]` is `(*this)[j]`
     * where `j` is the `i`th index of `mask` such that `mask[i]` is `true`.
     *
     * @param mask The mask containing `true` at indices to pick and
     * `false` and indices to ignore.
     * @throws std::runtime_error if `mask.size() != this->size()`.
     */
    inline Array<T> operator[](const Array<bool>& mask) const {
        if(size() != mask.size())  [[unlikely]]
            throw std::runtime_error("Size mismatch");
        return (*this)[where(mask)];
    }

    /**
     * @brief Fill `this` by copying items of `other`.
     *
     * All elements are copy-constructed.
     * Array does not define a copy operator, use this method instead.
     *
     * @param other The array to copy items from.
     * @throws std::runtime_error if `other.size() != this->size()`.
     */
    inline void assign(const Array<T>& other) {
        if(size() != other.size())  [[unlikely]]
            throw std::runtime_error("Size mismatch");
        fill_copy(other.data, data, size());
    }

    /**
     * @brief Replace every item of `this` by a copy of `value`.
     *
     * @param value The value to write to every item.
     */
    inline void assign(T value) {
        for(size_t i{0}; i < size(); ++i)
            data[i] = value;
    }

    /**
     * @brief Ensure that the items of `this` are contiguous in memory.
     *
     * This is always the case.
     */
    inline void ensure_contiguous() const {
        ;  // nop
    }

    /**
     * @brief Get the size of the array.
     */
    inline size_t size() const {
        return n;
    }

protected:
    T* data;
    size_t n;
    bool owns_data;

    static inline void fill_copy(const T* const src, T* dest, size_t n) {
        for(size_t i{0}; i < n; ++i)
            dest[i] = src[i];
    }

    void fill_copy(Array<T>& copy) const {
        assert(copy.size() == this->size());
        ensure_contiguous();
        copy.ensure_contiguous();  // Should always be ok, but just to make sure
        fill_copy(this->data, copy.data, size());
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
};


/********** Utils **********/

/**
 */
template <typename T=size_t>
inline Array<T> ones(size_t size) {
    return Array<T>(size, static_cast<T>(1));
}

/**
 * @relates Array
 *
 * @brief Create an array containing the indices of mask that are `true`.
 *
 * Create an array `arr` where `arr[i]` is the `ith` index `j` such that
 * `mask[j]` is `true`.
 *
 * Note that the returned array is necessarily increasing.
 *
 * @param mask The array to pick indices from.
 */
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

/**
 * @relates Array
 *
 * @brief Create an array containing the items [beg, beg+1, .., end-1] = [beg, end).
 *
 * @tparam T the type of the array. Defaults to size_t.
 * @param beg The first value of the range.
 * @param end The first value to not be in the range.
 */
template <typename T=size_t>
inline Array<T> range(size_t beg, size_t end) {
    Array<T> ret(end-beg);
    for(size_t i{0}; i < ret.size(); ++i)
        ret[i] = static_cast<T>(beg+i);
    return ret;
}


/**
 * @relates Array
 *
 * @brief Compute sum (array[i] * weights[i]).
 */
template <typename T>
inline T weighted_sum(const Array<T>& array, const Array<T>& weights) {
    assert(array.size() == weights.size());
    T ret{0};
    for(size_t i{0}; i < array.size(); ++i)
        ret += array[i]*weights[i];
    return ret;
}

/**
 * @relates Array
 *
 * @brief Compute sum array[i] where the sum is performed over type `T`,
 * even though items of `array` are of type `U`.
 */
template <typename T, typename U>
requires(std::is_convertible_v<U, T>)
inline T _typed_sum(const Array<U>& array) {
    T ret{0};
    for(size_t i{0}; i < array.size(); ++i)
        ret += array[i];
    return ret;
}

/**
 * @relates Array
 *
 * @brief Compute sum array[i].
 */
template <typename T>
inline T sum(const Array<T>& array) {
    return _typed_sum<T, T>(array);
}

/**
 * @relates Array
 *
 * Compute the mean of an Array by casting every item to a floating-point type.
 *
 * @tparam T The type of the array.
 * @tparam Float The floating-point type to convert items of array to.
 * @param array The array whose mean is computed.
 */
template <typename T, std::floating_point Float=double>
inline Float mean(const Array<T>& array) {
    if(array.size() == 0) [[unlikely]]
        return Float(0);
    return _typed_sum<Float, T>(array) / static_cast<Float>(array.size());
}

/**
 * @relates Array
 *
 * @brief Compute the weighted mean of an array.
 *
 * See mean() and weighted_sum().
 */
template <std::floating_point Float>
inline Float weighted_mean(
        const Array<Float>& array, const Array<Float>& weights) {
    if(array.size() == 0) [[unlikely]]
        return Float(0);
    Float sum{0};
    Float den{0};
    assert(array.size() == weights.size());
    for(size_t i{0}; i < array.size(); ++i) {
        sum += array[i]*weights[i];
        den += weights[i];
    }
    return sum / den;
}

/**
 * @relates Array
 *
 * @brief Compute sum (array[i] * weights[i]) over indices `i` such that
 * `array[i] == value`.
 */
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

/**
 * @relates Array
 *
 * @brief Get the number of unique values in `sorted_array`.
 *
 * WARNING: `sorted_array` is expected to be sorted.
 * Undefined behaviour if that is not the case.
 */
template <typename T>
static inline size_t nb_unique(const Array<T>& sorted_array) {
    size_t ret{1};
    for(size_t i{1}; i < sorted_array.size(); ++i)
        if(sorted_array[i] != sorted_array[i-1])
            ++ret;
    return ret;
}

/**
 * @relates Array
 *
 * @brief Compute the cumulative sum of `array`, i.e. `cumsum(array)[i]` is
 * sum array[j] over 0 <= j <= i.
 */
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
        os << x << ", ";
    return os;
}

}  // Cart::

#endif
