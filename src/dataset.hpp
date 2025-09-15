#ifndef CART_DATASET_HPP
#define CART_DATASET_HPP

#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <vector>

#include "array.hpp"
#include "random.hpp"
#include "utils.hpp"

namespace Cart {
template <typename Float>
class Dataset final {
public:
    Dataset(const Dataset&) = delete;
    Dataset(Dataset&& other):
            nb_obs{other.nb_obs}, nb_cols{other.nb_cols},
            _X(std::move(other._X)), _y(std::move(other._y)),
            _p(std::move(other._p)), _w(std::move(other._w)),
            __modalities(std::move(other._modalities)),
            _modalities{__modalities},
            _cache_sorted(std::move(other._cache_sorted)) {
    }
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p, Array<Float>&& w,
                std::vector<std::vector<std::string>>&& modalities):
            nb_obs{y.size()}, nb_cols{X.size() / y.size()},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(std::move(w)),
            __modalities(std::move(modalities)),
            _modalities{__modalities},
            _cache_sorted() {
    }
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p,
                std::vector<std::vector<std::string>>&& modalities):
            nb_obs{y.size()}, nb_cols{X.size() / y.size()},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(0),
            __modalities(std::move(modalities)),
            _modalities{__modalities},
            _cache_sorted() {
    }
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p, Array<Float>&& w,
                std::vector<std::vector<std::string>>& modalities):
            nb_obs{y.size()}, nb_cols{X.size() / y.size()},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(std::move(w)),
            __modalities(), _modalities{modalities} {
    }
    Dataset(Array<Float>&& X, Array<Float>&& y, Array<bool>&& p,
                std::vector<std::vector<std::string>>& modalities):
            nb_obs{y.size()}, nb_cols{X.size() / y.size()},
            _X(std::move(X)), _y(std::move(y)), _p(std::move(p)), _w(0),
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

    bool not_all_equal(int col_idx) const {
        const Float* Xj{get_feature_vector_ptr(col_idx)};
        Float entry{*Xj};
        for(size_t i{1}; i < nb_obs; ++i)
            if(Xj[i] != entry)
                return true;
        return false;
    }

    inline auto get_feature_vector(size_t col_idx, bool copy=false) const {
        return Array<Float>(
            const_cast<Dataset<Float>*>(this)->get_feature_vector_ptr(col_idx),
            nb_obs, copy
        );
    }

    inline size_t size() const {
        return nb_obs;
    }
    inline size_t nb_features() const {
        return nb_cols;
    }
    inline const Array<Float>& get_X() const {
        return _X;
    }
    inline const Array<Float>& get_y() const {
        return _y;
    }
    inline const Array<bool>& get_p() const {
        return _p;
    }
    inline const Array<Float>& get_w() const {
        return _w;
    }
    inline bool is_categorical(size_t j) const {
        return not _modalities[j].empty();
    }
    inline bool is_weighted() const {
        return _w.size() > 0;
    }

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
            Array<size_t> sorted_indices{argsort(Xj)};
            cache_sorted[j] = {
                std::move(Xj[sorted_indices]),
                std::move(_y[sorted_indices]),
                std::move(_p[sorted_indices]),
                (is_weighted() ? std::move(_w[sorted_indices]) : decltype(_w)()),
                decltype(sorted_indices)()
            };
            // Make sure we don't invalidate sorted_indices
            std::get<4>(cache_sorted[j]) = std::move(sorted_indices);
            cached[j] = true;
        }
        return _cache_sorted[j];
    }

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
    inline Dataset<Float>* sample(size_t k, bool replace) const {
        Array<size_t> indices{Random::choice(size(), k, replace)};
        assert(k == indices.size());
        return at(indices);
    }
    inline size_t get_nb_unique_modalities(size_t j) const {
        return nb_unique(sorted(get_feature_vector(j)));
    }
    inline const std::string& ith_modality_of(size_t i, size_t j) const {
        return _modalities.at(j).at(i);
    }

    inline std::pair<Dataset<Float>*, Dataset<Float>*> split(
            double frac, bool shuffle) const {
        auto size1{static_cast<size_t>(size() * frac)};
        auto indices{range(0, size())};
        if(shuffle)
            Random::permutation(indices);
        assert(size1 < size());
        auto left_indices{indices.view(0, size1)};
        auto right_indices{indices.view(size1, size())};
        assert(left_indices.size() + right_indices.size() == size());
        return std::make_pair(
            at(left_indices),
            at(right_indices)
        );
    }
private:
    size_t nb_obs;
    size_t nb_cols;
    Array<Float> _X;  // Stored column-wise! (Fortran-style)
    Array<Float> _y;
    Array<bool> _p;
    Array<Float> _w;
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
