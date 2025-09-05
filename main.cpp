#include <iostream>

#include "config.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "tree.hpp"

typedef double Float;
int main() {
    auto _data{Cart::Dataset<double>::load_from("./dataset_numerical.cartbin")};
    auto& data{*_data.at(Cart::range(0, 10'000))};
    std::cout << data.size() << '\n';
    Cart::TreeConfig config;
    config.interaction_depth = 10;
    config.minobs = 10;
    using Float = double;
    using Loss = Cart::Loss::LorenzCurve<Float, false>;
    // using Loss = Cart::Loss::PoissonDeviance<Float>;
    Cart::Regression::BaseRegressionTree<Float, Loss> tree(config);
    tree.fit(data);
    return 0;
}
