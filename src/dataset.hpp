#ifndef CART_DATASET_HPP
#define CART_DATASET_HPP

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "array.hpp"
#include "random.hpp"
#include "sort.hpp"
#include "utils.hpp"

namespace Cart {
namespace {
template <std::floating_point Float>
static inline size_t __nb_f__(const Array<Float>& X, const Array<Float>& y) {
    return y.size() > 0
        ? X.size() / y.size()
        : 0;
}
}

/**
 * @brief Wrapper for the (train|test|validation) sets.
 *
 * @tparam Float the type of data to store (typically float or double).
 *
 * A Dataset contains:
 * 1. `X` (stored as a concatenation of columns);
 * 2. `y`;
 * 3. `w` (the weights of different instances);
 * 4. `p` (discriminative covariate used as fairness criterion [UNUSED,TODO]).
 *
 * A Dataset can contain categorical covariates
 * (encoded as Float(0), Float(1), Float(2), etc.)
 * and a std::vector of std::string contains the mapping from the Float(i)
 * to the name of the ith modality for that covariate.
 */
template <typename Float>
class Dataset final {
public:
    Dataset(const Dataset&) = delete;

    /**
     * @brief Move constructor.
     */
    Dataset(Dataset&& other):
            nb_obs{other.nb_obs}, nb_cols{other.nb_cols},
            _X(std::move(other._X)), _y(std::move(other._y)),
            _p(std::move(other._p)), _w(std::move(other._w)),
            sum_of_weights{sum(_w)},
            __modalities(std::move(other._modalities)),
            _modalities{__modalities},
            _cache_sorted(std::move(other._cache_sorted)) {
    }
    /**
     * @brief Unwrapped move constructor.
     */
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p, Array<Float>&& w,
                std::vector<std::vector<std::string>>&& modalities):
            nb_obs{y.size()}, nb_cols{__nb_f__(X, y)},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(std::move(w)),
            sum_of_weights{sum(_w)},
            __modalities(std::move(modalities)),
            _modalities{__modalities},
            _cache_sorted() {
    }

    /**
     * @brief Unwrapped unweighted move constructor.
     */
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p,
                std::vector<std::vector<std::string>>&& modalities):
            nb_obs{y.size()}, nb_cols{__nb_f__(X, y)},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(0),
            sum_of_weights{0},
            __modalities(std::move(modalities)),
            _modalities{__modalities},
            _cache_sorted() {
    }

    /**
     * @brief Unwrapped move constructor that keeps discrete covariate modalities untouched.
     */
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p, Array<Float>&& w,
                std::vector<std::vector<std::string>>& modalities):
            nb_obs{y.size()}, nb_cols{__nb_f__(X, y)},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(std::move(w)),
            sum_of_weights{sum(_w)},
            __modalities(), _modalities{modalities} {
    }

    /**
     * @brief Unwrapped unweighted move constructor that keeps discrete covariate modalities untouched.
     */
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p,
                std::vector<std::vector<std::string>>& modalities):
            nb_obs{y.size()}, nb_cols{__nb_f__(X, y)},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(0),
            sum_of_weights{0},
            __modalities(), _modalities{modalities}, _cache_sorted() {
    }

    ~Dataset() = default;
    Dataset& operator=(const Dataset&) = delete;
    Dataset& operator=(Dataset&&) = delete;

    void save_to(const std::string& path) const {
        std::ofstream outfile(
            path,
            std::ios::out | std::ios::binary | std::ios::app
        );
        // Header
        char buffer[16] = {};
        std::memcpy(buffer, "CART", 4);
        buffer[4] = static_cast<char>(sizeof(Float));
        buffer[5] = 1;  // Version
        outfile.write(buffer, 16);
        // Data
        _X.save_to(outfile);
        _y.save_to(outfile);
        _p.save_to(outfile);
        _w.save_to(outfile);
        size_t nb_covs{nb_cols};
        outfile.write(AS_CONSTCHARPTR(&nb_covs), ssizeof(nb_covs));
        for(const auto& modalities : _modalities) {
            auto nb_modalities{modalities.size()};
            outfile.write(AS_CONSTCHARPTR(&nb_modalities), ssizeof(nb_modalities));
            for(const auto& modality : modalities) {
                auto mod_size{static_cast<ssize_t>(modality.size())};
                outfile.write(AS_CONSTCHARPTR(&mod_size), ssizeof(mod_size));
                outfile.write(AS_CONSTCHARPTR(modality.data()), mod_size);
            }
        }
    }

    static Dataset<Float> load_from(const std::string& path) {
        std::ifstream infile(path, std::ios::in | std::ios::binary);
        char buffer[16] = {};
        infile.read(buffer, 16);
        if(std::strncmp(buffer, "CART", 4) != 0 or buffer[4] != sizeof(Float))
            throw std::runtime_error("WRONG HEADER");
        auto X{decltype(Dataset<Float>::_X)::load_from(infile)};
        auto y{decltype(Dataset<Float>::_y)::load_from(infile)};
        auto p{decltype(Dataset<Float>::_p)::load_from(infile)};
        auto w{decltype(Dataset<Float>::_w)::load_from(infile)};
        decltype(Dataset<Float>::__modalities) modalities;
        size_t nb_covs;
        infile.read(AS_CHARPTR(&nb_covs), ssizeof(nb_covs));
        modalities.resize(nb_covs);
        for(size_t i{0}; i < nb_covs; ++i) {
            size_t nb_modalities;
            infile.read(AS_CHARPTR(&nb_modalities), ssizeof(nb_modalities));
            for(size_t j{0}; j < nb_modalities; ++j) {
                ssize_t mod_size;
                infile.read(AS_CHARPTR(&mod_size), ssizeof(mod_size));
                modalities[i].emplace_back(mod_size+1, '\0');
                infile.read(modalities[i].back().data(), mod_size);
            }
        }
        return Dataset<Float>(
            std::move(X), std::move(y), std::move(p),
            std::move(w), std::move(modalities)
        );
    }

    /**
     * @brief Test whether or not all samples have the same value for a given covariate.
     *
     * @param col_idx The index of the covariate.
     */
    bool not_all_equal(int col_idx) const {
        const Float* Xj{get_feature_vector_ptr(col_idx)};
        Float entry{*Xj};
        for(size_t i{1}; i < nb_obs; ++i)
            if(Xj[i] != entry)
                return true;
        return false;
    }

    /**
     * @brief Get an array containing the values of a given covariate.
     *
     * @param col_idx The index of the covariate.
     * @param copy `true` to make a copy, `false` to make a view.
     */
    inline auto get_feature_vector(size_t col_idx, bool copy=false) const {
        return Array<Float>(
            const_cast<Dataset<Float>*>(this)->get_feature_vector_ptr(col_idx),
            nb_obs, copy
        );
    }

    /**
     * @brief Get the weighted size of the dataset.
     *
     * If the dataset is unweighted, this is equivalent to Dataset::size.
     * If the dataset is weighted, this corresponds to the sum of all the weights.
     */
    inline Float weighted_size() const {
        if(is_weighted())
            return sum_of_weights;
        else
            return static_cast<Float>(size());
    }

    /**
     * @brief Get the number of samples.
     */
    inline size_t size() const {
        return nb_obs;
    }

    /**
     * @brief Alias of Dataset::nb_covariates;
     */
    inline size_t nb_features() const {
        return nb_covariates();
    }

    /**
     * @brief Get the number of covariates in the dataset.
     */
    inline size_t nb_covariates() const {
        return nb_cols;
    }

    /**
     * @brief Get the array containing the sample covariates.
     *
     * NOTE: The array is stored columnwise.
     */
    inline const Array<Float>& get_X() const {
        return _X;
    }

    /**
     * @brief Get the array of ground truth.
     */
    inline const Array<Float>& get_y() const {
        return _y;
    }

    /**
     * @brief Get the array of the discriminative covariate.
     */
    inline const Array<bool>& get_p() const {
        return _p;
    }

    /**
     * @brief Get the array of weights.
     *
     * If the dataset is unweighted, this array is empty (and points to nullptr).
     */
    inline const Array<Float>& get_w() const {
        return _w;
    }

    /**
     * @brief Get whether or not some covariate is categorical.
     *
     * @param j The index of the covariate.
     */
    inline bool is_categorical(size_t j) const {
        return not _modalities[j].empty();
    }

    /**
     * @brief Get whether or not the dataset is weighted.
     */
    inline bool is_weighted() const {
        return _w.size() > 0;
    }

    /**
     * @brief Get a permutation of the dataset where the feature covariates
     * are sorted according to some covariate.
     *
     * The output dataset contains the exact same values as `this`,
     * the indices stay consistent for `X`, `y`, `p` and `w`
     * but are not necessarily the same as indices in `this`.
     *
     * Conceptually, if `this` represents a dataset obtained by the following matrix
     * `[X | y | p | w]`, then `sorted_Xypw(j)` represents the same dataset,
     * just with some permutation of the rows of the matrix.
     * The samples stay the same!
     *
     * This is useful in some situations, e.g. when looking at possible _splits_
     * for the covariate `j`: if the dataset is sorted, a linear traversal of
     * the dataset is sufficient.
     *
     * NOTE: The sorted datasets are cached.
     * This can therefore be heavy w.r.t. memory usage.
     * Use carefully.
     *
     * @param j The index of the covariate to sort along.
     * @return a std::tuple containing:
     * 1. X columnwise, (sorted by covariate j)
     * 2. y
     * 3. p
     * 4. w
     * 5. indices, the argsort of X_j.
     */
    inline const auto& sorted_Xypw(size_t j) const {
        auto& cache_sorted{const_cast<Dataset<Float>*>(this)->_cache_sorted};
        auto& cached{const_cast<Dataset<Float>*>(this)->_cached};
        if(cache_sorted.size() == 0) [[unlikely]] {
            cached.resize(nb_features());
            cache_sorted.resize(nb_features());
            for(size_t j{0}; j < nb_features(); ++j)
                cached[j] = false;
        }
        assert(_cache_sorted.size() > j);
        if(not cached[j]) [[unlikely]] {
            Array<Float> Xj{get_feature_vector(j, false)};
            Sort::SortingAlgorithm method{
                is_categorical(j)
                ? Sort::SortingAlgorithm::QUICKSORT_3WAY
                : Sort::SortingAlgorithm::MERGESORT
            };
            Array<size_t> sorted_indices{argsort(Xj, method)};
            std::get<0>(cache_sorted[j]) = std::move(Xj[sorted_indices]);
            std::get<1>(cache_sorted[j]) = std::move(_y[sorted_indices]);
            std::get<2>(cache_sorted[j]) = std::move(_p[sorted_indices]);
            if(is_weighted())
                std::get<3>(cache_sorted[j]) = std::move(_w[sorted_indices]);
            std::get<4>(cache_sorted[j]) = std::move(sorted_indices);
            cached[j] = true;
        }
        return _cache_sorted[j];
    }

    /**
     * @brief Allocate a new Dataset containing only the samples at positions
     * where the mask is `true`.
     *
     * WARNING: The returned dataset is allocated using new
     * and must be freed using delete.
     *
     * See @ref Array_at_mask "Array::operator[](const Array<bool>&) const".
     */
    inline Dataset<Float>* at(const Array<bool>& mask) const {
        auto indices{where(mask)};
        return at(indices);
    }

    /**
     * @brief Allocate a new Dataset containing only the samples
     * whose indices are provided (possible duplicates).
     *
     * WARNING: The returned dataset is allocated using new
     * and must be freed using delete.
     *
     * See @ref at(const Array<bool>&) const and
     * \ref Array_at_indices "Array::operator[](const Array<size_t>&) const".
     */
    inline Dataset<Float>* at(const Array<size_t>& indices) const {
        Array<Float> newX(indices.size() * nb_features());
        for(size_t j{0}; j < nb_features(); ++j) {
            const Float * const ptr{get_feature_vector_ptr(j)};
            for(size_t i{0}; i < indices.size(); ++i)
                newX[j * indices.size() + i] = ptr[indices[i]];
        }
        if(is_weighted()) {
            return new Dataset<Float>(
                std::move(newX),
                std::move(_y[indices]),
                std::move(_p[indices]),
                std::move(_w[indices]),
                _modalities
            );
        } else {
            return new Dataset<Float>(
                std::move(newX),
                std::move(_y[indices]),
                std::move(_p[indices]),
                decltype(_w)(),
                _modalities
            );
        }
    }

    /**
     * @brief Sample a subdataset.
     *
     * @param k The number of samples to pick.
     * @param replace `true` to allow replacements, `false` to not allow them.
     */
    inline Dataset<Float>* sample(size_t k, bool replace) const {
        Array<size_t> indices{Random::choice(size(), k, replace)};
        return at(indices);
    }

    /**
     * @brief Get the number of unique modalities of a given covariate.
     *
     * @param j The index of the covariate.
     */
    inline size_t get_nb_unique_modalities(size_t j) const {
        return nb_unique(Sort::sorted(get_feature_vector(j)));
    }

    /**
     * @brief Get the name of a modality for a given covariate.
     *
     * Undefined behaviour if j is not a categorical, no check is performed
     * or if i is not a valid modality.
     * (Probably a raised std::out_of_range in both situations.)
     *
     * @param i The encoded value of the modality.
     * @param j The index of the covariate.
     */
    inline const std::string& ith_modality_of(size_t i, size_t j) const {
        return _modalities.at(j).at(i);
    }

    /**
     * @brief Split the Dataset in two parts of specified size.
     *
     * @param frac The fraction (in [0, 1]) of the dataset to end up in the first one.
     * @param shuffle `true` to pick the two subdatasets at random,
     *   `false` to take them in order of indices.
     */
    inline std::pair<Dataset<Float>*, Dataset<Float>*> split(
            double frac, bool shuffle) const {
        auto size1{static_cast<size_t>(size() * frac)};
        auto indices{range(0, size())};
        if(shuffle)
            Random::permutation(indices);
        auto left_indices{indices.view(0, size1)};
        auto right_indices{indices.view(size1, size())};
        return std::make_pair(
            at(left_indices),
            at(right_indices)
        );
    }

    /**
     * @brief Split the Dataset in two parts depending on the value of a
     * non-categorical covariate.
     *
     * If the value of the covariate is at most `threshold`,
     * then the sample ends up in the first subdataset; and if the value
     * is strictly greater that `threshold`, then the sample
     * ends up in the second one.
     *
     * @param j The index of the covariate.
     * @param threshold The threshold value to split with.
     */
    inline std::pair<Dataset<Float>*, Dataset<Float>*> split_on(
            int j, Float threshold) const {
        assert(not is_categorical(j));
        auto [Xj, y, p, w, indices] = sorted_Xypw(j);
        auto it{std::lower_bound(Xj.cbegin(), Xj.cend(), threshold)};
        assert(it != Xj.cend());
        auto idx{static_cast<size_t>(std::distance(Xj.cbegin(), it))};
        return {
            at(indices.view(0, idx)),
            at(indices.view(idx, indices.size()))
        };
    }

    /**
     * @brief Split the Dataset in two parts depending on the value of a
     * categorical covariate.
     *
     * See split_on(int, Float) const.
     *
     * The argument `mask` is such that if `mask & (1 << i)`, then samples
     * whose modality of the covariate `j` is the `i`th modality go to the
     * first subdataset; otherwise they go to the second one.
     *
     *
     * @param j The index of the covariate.
     * @param mask A bitwise selection of the modalities to end up in the
     *   first subdataset.
     */
    inline std::pair<Dataset<Float>*, Dataset<Float>*> split_on(
            int j, uint64_t mask) const {
        assert(is_categorical(j));
        const Array<Float>& Xj{get_feature_vector(j)};
        Array<bool> go_left(size(), false);
        Array<bool> go_right(size(), false);
        for(size_t i{0}; i < size(); ++i) {
            uint64_t flag{1ull << static_cast<int>(Xj[i])};
            if(mask & flag)
                go_left[i] = true;
            else
                go_right[i] = true;
        }
        return {
            at(go_left),
            at(go_right)
        };
    }
private:
    size_t nb_obs;
    size_t nb_cols;
    Array<Float> _X;  // Stored column-wise! (Fortran-style)
    Array<Float> _y;
    Array<bool> _p;
    Array<Float> _w;
    Float sum_of_weights;
    std::vector<std::vector<std::string>> __modalities;
    std::vector<std::vector<std::string>>& _modalities;
    std::vector<
        std::tuple<
            Array<Float>,   // Xj
            Array<Float>,   // y
            Array<bool>,    // p
            Array<Float>,   // w
            Array<size_t>   // indices
        >
    > _cache_sorted;
    std::vector<bool> _cached;


    inline Float* get_feature_vector_ptr(size_t col_idx) {
        _X.ensure_contiguous();
        return &*_X.begin() + nb_obs*col_idx;
    }
    inline const Float* get_feature_vector_ptr(size_t col_idx) const {
        _X.ensure_contiguous();
        return &*_X.begin() + nb_obs*col_idx;
    }
};
}  // Cart::

#endif
