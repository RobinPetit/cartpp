#ifndef __PYCART__
#define __PYCART__

#include "dataset.hpp"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "tree.hpp"
#include <concepts>
#include <type_traits>

enum class __FloatingPoint {
    FLOAT32,
    FLOAT64
};
enum class __Loss {
    MSE,
    POISSON_DEVIANCE
};

template <std::floating_point Float>
using CartFloat = Cart::Dataset<Float>;
template <std::floating_point Float>
using CartSplitChoice = Cart::SplitChoice<Float>;
template <std::floating_point Float, typename LossType>
using CartRegressionTree = Cart::Regression::BaseRegressionTree<Float, LossType>;

template <typename Float>
static inline Cart::Dataset<Float>* _make_dataset(
        Float* Xptr, Float* yptr, bool* pptr,
        std::vector<std::vector<std::string>>&& modalities,
        size_t n, size_t m) {
    return new Cart::Dataset<Float>(
        std::move(Cart::Array<Float>(Xptr, n*m)),
        std::move(Cart::Array<Float>(yptr, n)),
        std::move(Cart::Array<bool>(pptr, n)),
        std::move(modalities)
    );
}
template <typename Float>
static inline void _del_dataset(void* ptr) {
    auto _ptr{static_cast<Cart::Dataset<Float>*>(ptr)};
    delete _ptr;
}
template <typename Float>
static inline void _save_dataset(void* ptr, const char* path) {
    static_cast<Cart::Dataset<Float>*>(ptr)->save_to(path);
}
template <typename Float>
static inline bool _is_categorical(void* ptr, size_t j) {
    return static_cast<Cart::Dataset<Float>*>(ptr)->is_categorical(j);
}

typedef float CART_FLOAT32;
typedef double CART_FLOAT64;

#define NBT(F, L) Cart::Regression::NodeBasedRegressionTree<F, L<F>>

static inline void CALL_FIT(void* tree, void* dataset,
                            __FloatingPoint fp, __Loss loss) {
#define CALL(F, L) static_cast<NBT(F, L)*>(tree)->fit( \
        *static_cast<Cart::Dataset<F>*>(dataset))
    if(loss == __Loss::MSE) {
        if(fp == __FloatingPoint::FLOAT32) {
            CALL(CART_FLOAT32, Cart::Loss::MeanSquaredError);
        } else if(fp == __FloatingPoint::FLOAT64) {
            CALL(CART_FLOAT64, Cart::Loss::MeanSquaredError);
        } else {
            // Unreachable
        }
    } else if(loss == __Loss::POISSON_DEVIANCE) {
        if(fp == __FloatingPoint::FLOAT32) {
            CALL(CART_FLOAT32, Cart::Loss::PoissonDeviance);
        } else if(fp == __FloatingPoint::FLOAT64) {
            CALL(CART_FLOAT64, Cart::Loss::PoissonDeviance);
        } else {
            // Unreachable
        }
    } else {
        // unreachable
    }
#undef CALL
}
static inline void* CREATE_TREE(Cart::TreeConfig* config,
                                __FloatingPoint fp, __Loss loss) {
#define ALLOC(F, L) new NBT(F, L)(*config)
    void* ret{nullptr};
    if(loss == __Loss::MSE) {
        if(fp == __FloatingPoint::FLOAT32) {
            ret = ALLOC(CART_FLOAT32, Cart::Loss::MeanSquaredError);
        } else if(fp == __FloatingPoint::FLOAT64) {
            ret = ALLOC(CART_FLOAT64, Cart::Loss::MeanSquaredError);
        } else {
            // Unreachable
        }
    } else if(loss == __Loss::POISSON_DEVIANCE) {
        if(fp == __FloatingPoint::FLOAT32) {
            ret = ALLOC(CART_FLOAT32, Cart::Loss::PoissonDeviance);
        } else if(fp == __FloatingPoint::FLOAT64) {
            ret = ALLOC(CART_FLOAT64, Cart::Loss::PoissonDeviance);
        } else {
            // Unreachable
        }
    } else {
        // unreachable
    }
    return ret;
#undef ALLOC
}
static inline void DELETE_TREE(void* tree, __FloatingPoint fp, __Loss loss) {
#define DELETE(F, L) delete static_cast<NBT(F, L)*>(tree)
    if(loss == __Loss::MSE) {
        if(fp == __FloatingPoint::FLOAT32) {
            DELETE(CART_FLOAT32, Cart::Loss::MeanSquaredError);
        } else if(fp == __FloatingPoint::FLOAT64) {
            DELETE(CART_FLOAT64, Cart::Loss::MeanSquaredError);
        } else {
            // Unreachable
        }
    } else if(loss == __Loss::POISSON_DEVIANCE) {
        if(fp == __FloatingPoint::FLOAT32) {
            DELETE(CART_FLOAT32, Cart::Loss::PoissonDeviance);
        } else if(fp == __FloatingPoint::FLOAT64) {
            DELETE(CART_FLOAT64, Cart::Loss::PoissonDeviance);
        } else {
            // Unreachable
        }
    } else {
        // unreachable
    }
#undef DELETE
}
#undef NBT

static size_t CART_DEFAULT{std::numeric_limits<size_t>::max()};

/* Numpy array as Cart::Array view */

#include <numpy/ndarrayobject.h>

template <typename T>
static inline int __GET_NP_TYPE() {
    int typenum{0};
    if constexpr(std::is_same_v<T, bool>) {
        static_assert(sizeof(bool) == sizeof(npy_bool), "Wrong boolean type");
        typenum = NPY_BOOL;
    } else if constexpr(std::is_same_v<T, CART_FLOAT32>) {
        typenum = NPY_FLOAT32;
    } else if constexpr(std::is_same_v<T, CART_FLOAT64>) {
        typenum = NPY_FLOAT64;
    } else {
        static_assert(false, "Unable to convert given type no np.ndarray");
    }
    return typenum;
}

template <typename T>
static inline PyObject* __Cart_Array_to_ndarray_2d(
            const Cart::Array<T>& array, int rows, int cols) {
    npy_intp dims[2]{rows, cols};
    return PyArray_SimpleNewFromData(
        2, dims, __GET_NP_TYPE<T>(),
        const_cast<void*>(static_cast<const void*>(&*array.begin()))
    );
}

template <typename T>
static inline PyObject* __Cart_Array_to_np_ndarray_view(const Cart::Array<T>& array) {
    auto size{static_cast<npy_intp>(array.size())};
    return PyArray_SimpleNewFromData(
        1, &size, __GET_NP_TYPE<T>(),
        const_cast<void*>(static_cast<const void*>(&*array.begin()))
    );
}

static inline PyObject* __Cart_Arraybool_to_np(const Cart::Array<bool>& array) {
    return __Cart_Array_to_np_ndarray_view(array);
}

static inline PyObject* __Cart_Arrayfloat32_to_np(const Cart::Array<CART_FLOAT32>& array) {
    return __Cart_Array_to_np_ndarray_view(array);
}

static inline PyObject* __Cart_Arrayfloat64_to_np(const Cart::Array<CART_FLOAT64>& array) {
    return __Cart_Array_to_np_ndarray_view(array);
}

template <std::floating_point Float>
static inline PyObject* __Dataset_get_X(const void* ptr) {
    auto dataset{static_cast<const Cart::Dataset<Float>*>(ptr)};
    auto const& array{dataset->get_X()};
    return __Cart_Array_to_ndarray_2d(
        array, static_cast<int>(dataset->nb_features()), static_cast<int>(dataset->size())
    );
}
#define _GET_ARRAY(p, m) \
    __Cart_Array_to_np_ndarray_view( \
        static_cast<const Cart::Dataset<Float>*>(p)->m() \
    )
#define _DEFINE_DATASET_GET_(a) \
    template <std::floating_point Float> \
    static inline PyObject* __Dataset_get_ ## a (const void* d) { \
        return _GET_ARRAY(d, get_ ## a); \
    }

_DEFINE_DATASET_GET_(y)
_DEFINE_DATASET_GET_(p)
_DEFINE_DATASET_GET_(w)

#endif
