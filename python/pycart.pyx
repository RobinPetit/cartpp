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

import threading
from joblib import Parallel as _Parallel, delayed

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
        bool bootstrap
        double bootstrap_frac
        bool bootstrap_replacement
        bool exact_splits
        NodeSelector split_type
        size_t max_depth
        size_t interaction_depth
        size_t minobs
        bool verbose

cdef extern from "_pycart.hpp" nogil:
    void* _make_dataset[T](T*, T*, bool*, vector[vector[string]], size_t, size_t)
    void _del_dataset[T](void*)
    void _save_dataset[T](void*, const char*)
    bool _is_categorical[T](void*, size_t)
    np.ndarray __Dataset_get_X[T](const void*)
    np.ndarray __Dataset_get_y[T](const void*)
    np.ndarray __Dataset_get_w[T](const void*)
    np.ndarray __Dataset_get_p[T](const void*)

    cdef enum class __FloatingPoint(int):
        FLOAT32,
        FLOAT64

    cdef enum class __Loss(int):
        MSE,
        POISSON_DEVIANCE,
        NON_CROSSING_LORENZ,
        CROSSING_LORENZ

    void CALL_FIT_TREE(
            void* tree, void* dataset,
            __FloatingPoint fp, __Loss loss
    )
    void CALL_PREDICT_TREE(
            const void* tree, void* X, void* out, int n, int nb_dim,
            __FloatingPoint fp, __Loss loss
    )
    void CALL_CREATE_TREE(
            void** tree, TreeConfig* config,
            __FloatingPoint fp, __Loss loss
    )
    void CALL_DELETE_TREE(
            void* tree,
            __FloatingPoint fp, __Loss loss
    )
    void CALL_GET_NB_INTERNAL_NODES_TREE(
            void* tree, size_t* size,
            __FloatingPoint fp, __Loss loss
    )
    void CALL_GET_FEATURE_IMPORTANCE_TREE(
            void* tree, void* array,
            __FloatingPoint fp, __Loss loss
    )
    cdef size_t CART_DEFAULT

    void _extract_lorenz_curves[T](void* tree, np.float64_t* out)


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
        cdef size_t N = <size_t>(y.shape[0])
        cdef size_t nb_features = <size_t>(_X.shape[1])
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
        X = np.ascontiguousarray(X)
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

    def get_X(self) -> np.ndarray:
        if self.dtype is np.float32:
            return __Dataset_get_X[CART_FLOAT32](self.ptr).T
        elif self.dtype is np.float64:
            return __Dataset_get_X[CART_FLOAT64](self.ptr).T

    def get_y(self) -> np.ndarray:
        if self.dtype is np.float32:
            return __Dataset_get_y[CART_FLOAT32](self.ptr)
        elif self.dtype is np.float64:
            return __Dataset_get_y[CART_FLOAT64](self.ptr)

    def get_p(self) -> np.ndarray:
        if self.dtype is np.float32:
            return __Dataset_get_p[CART_FLOAT32](self.ptr)
        elif self.dtype is np.float64:
            return __Dataset_get_p[CART_FLOAT64](self.ptr)

    def get_w(self) -> np.ndarray:
        if self.dtype is np.float32:
            return __Dataset_get_w[CART_FLOAT32](self.ptr)
        elif self.dtype is np.float64:
            return __Dataset_get_w[CART_FLOAT64](self.ptr)

cdef class Config:
    cdef TreeConfig _config
    cdef __Loss _loss
    cdef type dtype
    cdef __FloatingPoint _fp

    AVAILABLE_LOSSES = ['mse', 'poisson', 'lorenz']

    def __init__(self, str loss, type dtype=np.float32, bool exact_splits=True,
                 str split_type='best', size_t max_depth=CART_DEFAULT,
                 size_t interaction_depth=CART_DEFAULT,
                 int minobs=1, crossing_lorenz: bool=False,
                 bootstrap: bool=False,
                 bootstrap_frac: float=1.,
                 bootstrap_replacement: bool=True,
                 verbose: bool=False):
        cdef str _loss = loss.lower().strip()
        assert _loss in Config.AVAILABLE_LOSSES, f"Unknown loss '{loss}'"
        if _loss == 'mse':
            self._loss = __Loss.MSE
        elif _loss == 'poisson':
            self._loss = __Loss.POISSON_DEVIANCE
        elif _loss == 'lorenz':
            if crossing_lorenz:
                self._loss = __Loss.CROSSING_LORENZ
            else:
                self._loss = __Loss.NON_CROSSING_LORENZ
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
        self._config.bootstrap = bootstrap
        self._config.bootstrap_frac = bootstrap_frac
        self._config.bootstrap_replacement = bootstrap_replacement
        self._config.verbose = verbose

cdef class RegressionTree:
    cdef void* _tree
    cdef Config config
    def __init__(self, config):
        self._tree = NULL
        self.config = config
        with nogil:
            CALL_CREATE_TREE(
                &self._tree, &self.config._config, self.config._fp, self.config._loss
            )
        assert self._tree != NULL

    def __dealloc__(self):
        with nogil:
            CALL_DELETE_TREE(self._tree, self.config._fp, self.config._loss)

    def fit(self, Dataset dataset):
        assert dataset.dtype == self.config.dtype
        with nogil:
            CALL_FIT_TREE(self._tree, dataset.ptr, self.config._fp, self.config._loss)

    def predict(self, np.ndarray X) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        cdef np.float32_t[:] _ret_32
        cdef np.float64_t[:] _ret_64
        cdef int n = <int>(X.shape[0])
        cdef int nb_dim = <int>(X.shape[1])
        cdef void* ptr = <void*><long long>(X.ctypes.data)
        if self.config._fp == __FloatingPoint.FLOAT32:
            _ret_32 = np.empty(n, dtype=np.float32)
            with nogil:
                CALL_PREDICT_TREE(
                    self._tree,
                    ptr, &_ret_32[0], n, nb_dim,
                    self.config._fp, self.config._loss,
                )
            return np.asarray(_ret_32)
        else:
            _ret_64 = np.empty(n, dtype=np.float64)
            with nogil:
                CALL_PREDICT_TREE(
                    self._tree,
                    ptr, &_ret_64[0], n, nb_dim,
                    self.config._fp, self.config._loss,
                )
            return np.asarray(_ret_64)
        raise ValueError()

    def get_nb_internal_nodes(self) -> int:
        cdef size_t ret
        with nogil:
            CALL_GET_NB_INTERNAL_NODES_TREE(
                self._tree, &ret,
                self.config._fp, self.config._loss
            )
        return ret

    def get_feature_importance(self, nb_features) -> np.ndarray:
        cdef np.float32_t[:] _ret32
        cdef np.float64_t[:] _ret64
        if self.config._fp == __FloatingPoint.FLOAT32:
            _ret32 = np.zeros(nb_features, dtype=np.float32)
            with nogil:
                CALL_GET_FEATURE_IMPORTANCE_TREE(
                    self._tree, &_ret32[0],
                    self.config._fp, self.config._loss
                )
            return np.asarray(_ret32)
        else:
            _ret64 = np.zeros(nb_features, dtype=np.float64)
            with nogil:
                CALL_GET_FEATURE_IMPORTANCE_TREE(
                    self._tree, &_ret64[0],
                    self.config._fp, self.config._loss
                )
            return np.asarray(_ret64)

    def get_lorenz_curves(self) -> np.ndarray:
        n = self.get_nb_internal_nodes()
        cdef np.float64_t[:] ret = np.empty((n+1)*(n+4), dtype=np.float64)
        if self.config._fp == __FloatingPoint.FLOAT32:
            _extract_lorenz_curves[CART_FLOAT32](self._tree, &ret[0])
        else:
            _extract_lorenz_curves[CART_FLOAT64](self._tree, &ret[0])
        return np.asarray(ret)

def Parallel(n_jobs):
    return _Parallel(n_jobs=n_jobs, backend='threading', prefer='threads')

def regressor_fit(callback, dataset):
    callback(dataset)

def regressor_predict(callback, X, out, lock):
    predictions = callback(X)
    with lock:
        out += predictions

def regressor_predict_incremental(callback, X, out, i):
    out[:, i] = callback(X)

def regressor_importance(callback, nb_features, out, lock):
    importances = callback(nb_features)
    with lock:
        out += importances

cdef class RandomForest:
    cdef list trees
    cdef int n_jobs
    cdef bint fitted
    cdef Config config
    def __init__(self, config, nb_trees: int, n_jobs: int):
        self.n_jobs = n_jobs
        self.config = config
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(RegressionTree)(
                config
            )
            for _ in range(nb_trees)
        )
        self.fitted = False

    def __len__(self):
        return len(self.trees)

    def fit(self, dataset):
        Parallel(n_jobs=self.n_jobs)(
            delayed(regressor_fit)(
                self.trees[i].fit, dataset
            )
            for i in range(len(self.trees))
        )
        self.fitted = True

    def predict(self, X):
        assert self.fitted
        if self.config._fp == __FloatingPoint.FLOAT32:
            dtype = np.float32
        else:
            dtype = np.float64
        assert X.dtype == dtype
        lock = threading.Lock()
        out = np.zeros(X.shape[0], dtype=dtype)
        Parallel(n_jobs=self.n_jobs)(
            delayed(regressor_predict)(
                self.trees[i].predict, X, out, lock
            )
            for i in range(len(self.trees))
        )
        out /= len(self.trees)
        return out

    def predict_incremental(self, X):
        assert self.fitted
        if self.config._fp == __FloatingPoint.FLOAT32:
            dtype = np.float32
        else:
            dtype = np.float64
        out = np.zeros((X.shape[0], len(self.trees)), dtype=np.float64)
        Parallel(n_jobs=self.n_jobs)(
            delayed(regressor_predict_incremental)(
                self.trees[i].predict, X, out, i
            )
            for i in range(len(self.trees))
        )
        np.cumsum(out, axis=1, out=out)
        column_indices = np.arange(1, out.shape[1] + 1)
        out /= column_indices
        return out

    def get_feature_importance(self, nb_features):
        assert self.fitted
        if self.config._fp == __FloatingPoint.FLOAT32:
            dtype = np.float32
        else:
            dtype = np.float64
        out = np.zeros(nb_features, dtype=dtype)
        lock = threading.Lock()
        Parallel(n_jobs=self.n_jobs)(
            delayed(regressor_importance)(
                self.trees[i].get_feature_importance, nb_features, out, lock
            )
            for i in range(len(self.trees))
        )
        out /= len(self.trees)
        return out
