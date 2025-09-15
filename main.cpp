#include <iostream>

#include "config.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "tree.hpp"

typedef double Float;
int main() {
    auto _data{Cart::Dataset<double>::load_from("./dataset_filtered_cat.cartbin")};
    auto [a, b] = _data.split(.5, false);
    auto& data{*_data.at(Cart::range(0, 100'000))};
    // auto& data{_data};
    std::cout << data.size() << '\n';
    std::cout << data.nb_features() << '\n';
    Cart::TreeConfig config;
    config.interaction_depth = 31;
    config.minobs = 100;
    config.verbose = true;
    using Float = double;
    using Loss = Cart::Loss::PoissonDeviance<Float>;
    // using Loss = Cart::Loss::CrossingLorenzCurveError<Float>;
    // using Loss = Cart::Loss::PoissonDeviance<Float>;
    Cart::Regression::BaseRegressionTree<Float, Loss> tree(config);
    tree.fit(data);
    delete a;
    delete b;
    if(&data != &_data)
        delete &data;
    return 0;
}
