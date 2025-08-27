# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3
# cython: linetrace=True
# cython: language=c++
# cython: c_string_type=unicode, c_string_encoding=UTF8

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

cdef extern from "array.hpp" namespace "Cart" nogil:
    cdef cppclass Array[T]:
        pass

cdef extern from "splitter.hpp" namespace "Cart" nogil:
    cdef enum class NodeSelector(int):
        BEST_FIRST,
        DEPTH_FIRST

cdef extern from "tree.hpp" namespace "Cart" nogil:
    cdef struct TreeConfig:
        bool exact_splits
        NodeSelector split_type
        size_t max_depth
        size_t interaction_depth
        size_t minobs

cdef extern from * nogil:
    '''
#include "dataset.hpp"
#include "tree.hpp"

template <std::floating_point Float>
using CartFloat = Cart::Dataset<Float>;
template <std::floating_point Float>
using CartSplitChoice = Cart::SplitChoice<Float>;
template <std::floating_point Float, typename LossType>
using CartRegressionTree = Cart::Regression::BaseRegressionTree<Float, LossType>;

template <typename Float>
static inline Cart::Dataset<Float>* _make_dataset(
        Float* Xptr, Float* yptr, bool* pptr,
        // bool* cat,
        std::vector<std::vector<std::string>>&& modalities,
        size_t n, size_t m) {
    return new Cart::Dataset<Float>(
        std::move(Cart::Array<Float>(Xptr, n*m)),
        std::move(Cart::Array<Float>(yptr, n)),
        std::move(Cart::Array<bool>(pptr, n)),
        std::move(modalities)
        // std::shared_ptr<bool[]>(cat)
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

enum class __FloatingPoint {
    FLOAT32,
    FLOAT64
};
enum class __Loss {
    MSE,
    POISSON_DEVIANCE
};

typedef float CART_FLOAT32;
typedef double CART_FLOAT64;

#define NBL(F, L) Cart::Regression::NodeBasedRegressionTree<F, L<F>>
static inline void CALL_FIT(void* tree, void* dataset,
                            __FloatingPoint fp, __Loss loss) {
#define CALL(F, L) static_cast<NBL(F, L)*>(tree)->fit( \
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
#define ALLOC(F, L) new NBL(F, L)(*config)
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
#define DELETE(F, L) delete static_cast<NBL(F, L)*>(tree)
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
#undef NBL
template <typename T>
static inline T* __new_array__(size_t n) {
    return new T[n];
}
static size_t CART_DEFAULT{std::numeric_limits<size_t>::max()};
    '''
    void* _make_dataset[T](T*, T*, bool*, vector[vector[string]], size_t, size_t)
    void _del_dataset[T](void*)
    void _save_dataset[T](void*, const char*)
    bool _is_categorical[T](void*, size_t)

    cdef enum class __FloatingPoint(int):
        FLOAT32,
        FLOAT64

    cdef enum class __Loss(int):
        MSE,
        POISSON_DEVIANCE

    void CALL_FIT(void* tree, void* dataset, __FloatingPoint fp, __Loss loss)
    void* CREATE_TREE(TreeConfig* config, __FloatingPoint fp, __Loss loss)
    void DELETE_TREE(void* tree, __FloatingPoint fp, __Loss loss)
    # size_t get_nb_splitting_nodes(void*)
    T* __new_array__[T](size_t n)
    cdef size_t CART_DEFAULT


# TODO: make sure that sizeof(CART_FLOAT32) == 4 and sizeof(CART_FLOAT64) == 8
ctypedef float CART_FLOAT32
ctypedef double CART_FLOAT64
ctypedef ptrdiff_t CART_PTR_T

cdef class Dataset:
    cdef void* ptr
    cdef type dtype
    def __init__(self, np.ndarray[object, ndim=2] X, np.ndarray y,
                 np.ndarray p, dtype=np.float32):
        self.ptr = NULL
        self.dtype = dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError('Unknown dtype')
        assert set(map(float, np.unique(p))) == {0., 1.}, np.unique(p)
        cdef vector[vector[string]] modalities
        modalities.resize(X.shape[1])
        _X = self._create_X(X, modalities)
        y = np.ascontiguousarray(y.astype(self.dtype))
        p = np.ascontiguousarray(p.astype(np.uint8))
        cdef int N = y.shape[0]
        assert p.shape[0] == N
        assert _X.shape[0] == N, (N, tuple([_X.shape[0], _X.shape[1]]))
        cdef int nb_features = _X.shape[1]
        if self.dtype is np.float32:
            self.ptr = _make_dataset[CART_FLOAT32](
                <CART_FLOAT32*>(<CART_PTR_T>(_X.ctypes.data)),
                <CART_FLOAT32*>(<CART_PTR_T>(y.ctypes.data)),
                <bool*>(<CART_PTR_T>(p.ctypes.data)),
                move(modalities), N, nb_features
            )
        elif self.dtype is np.float64:
            self.ptr = _make_dataset[CART_FLOAT64](
                <CART_FLOAT64*>(<CART_PTR_T>(_X.ctypes.data)),
                <CART_FLOAT64*>(<CART_PTR_T>(y.ctypes.data)),
                <bool*>(<CART_PTR_T>(p.ctypes.data)),
                move(modalities), N, nb_features
            )

    def __dealloc__(self):
        if self.dtype is np.float32:
            _del_dataset[CART_FLOAT32](self.ptr)
        elif self.dtype is np.float64:
            _del_dataset[CART_FLOAT64](self.ptr)

    def save_to(self, path: str) -> None:
        if self.dtype is np.float32:
            _save_dataset[CART_FLOAT32](self.ptr, path)
        elif self.dtype is np.float64:
            _save_dataset[CART_FLOAT64](self.ptr, path)

    cdef np.ndarray _create_X(self, np.ndarray[object, ndim=2] X,
                              vector[vector[string]]& modalities):
        cdef np.ndarray _X = -np.ones_like(X, dtype=self.dtype, order='F')
        for j in range(X.shape[1]):
            x = X[:, j]
            if isinstance(x[0], str):
                self._labelize(_X[:, j], x, modalities[j])
                wrong_indices = np.where(_X[:, j] < 0)[0]
                assert np.all(_X[:, j] >= 0), X[wrong_indices, j]
            else:
                assert isinstance(x[0], (float, int))
                _X[:, j] = x[:]
        return _X

    cdef void _labelize(self, np.ndarray out, np.ndarray[object, ndim=1] in_,
                        vector[string]& modalities):
        cdef tuple uniques = np.unique(in_, return_index=True)
        # For reproducibility
        cdef np.ndarray unique_values = uniques[0][np.argsort(uniques[1])]
        modalities.reserve(unique_values.shape[0])
        cdef int counter = 0
        for i in range(unique_values.shape[0]):
            modalities.push_back(unique_values[i])
            indices = np.where(in_ == unique_values[i])[0]
            out[indices] = counter
            counter += 1

    def is_categorical(self, int j) -> bool:
        if self.dtype is np.float32:
            return _is_categorical[CART_FLOAT32](self.ptr, j)
        elif self.dtype is np.float64:
            return _is_categorical[CART_FLOAT64](self.ptr, j)

cdef class Config:
    cdef TreeConfig _config
    cdef __Loss _loss
    cdef type dtype
    cdef __FloatingPoint _fp

    AVAILABLE_LOSSES = ['mse', 'poisson']

    def __init__(self, str loss, type dtype=np.float32, bool exact_splits=True,
                 str split_type='best', size_t max_depth=CART_DEFAULT,
                 size_t interaction_depth=CART_DEFAULT,
                 int minobs=1):
        cdef str _loss = loss.lower().strip()
        assert _loss in Config.AVAILABLE_LOSSES, f"Unknown loss '{loss}'"
        if _loss == 'mse':
            self._loss = __Loss.MSE
        elif _loss == 'poisson':
            self._loss = __Loss.POISSON_DEVIANCE
        else:
            raise ValueError()
        self.dtype = dtype
        if dtype is np.float32:
            self._fp = __FloatingPoint.FLOAT32
        elif dtype is np.float64:
            self._fp = __FloatingPoint.FLOAT64
        else:
            raise ValueError()
        if split_type == 'best':
            self._config.split_type = NodeSelector.BEST_FIRST
        elif split_type == 'depth':
            self._config.split_type = NodeSelector.DEPTH_FIRST
        else:
            raise ValueError()
        self._config.exact_splits = exact_splits
        self._config.interaction_depth = interaction_depth
        self._config.max_depth = max_depth
        self._config.minobs = minobs

cdef class RegressionTree:
    cdef void* _tree
    cdef Config config
    def __init__(self, config):
        self._tree = NULL
        self.config = config
        self._tree = CREATE_TREE(
            &self.config._config, self.config._fp, self.config._loss
        )
        assert self._tree != NULL

    def __dealloc__(self):
        DELETE_TREE(self._tree, self.config._fp, self.config._loss)

    def fit(self, Dataset dataset):
        assert dataset.dtype == self.config.dtype
        CALL_FIT(self._tree, dataset.ptr, self.config._fp, self.config._loss)
