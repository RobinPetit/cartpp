#include <iostream>

#include "config.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "tree.hpp"

typedef double Float;
int main() {
    auto _data{Cart::Dataset<double>::load_from("./dataset_filtered_cat.cartbin")};
    auto& data{*_data.at(Cart::range(0, 100'000))};
    // auto& data{_data};
    std::cout << data.size() << '\n';
    std::cout << data.nb_features() << '\n';
    Cart::TreeConfig config;
    config.interaction_depth = 101;
    config.minobs = 10;
    using Float = double;
    using Loss = Cart::Loss::CrossingLorenzCurveError<Float>;
    // using Loss = Cart::Loss::PoissonDeviance<Float>;
    Cart::Regression::BaseRegressionTree<Float, Loss> tree(config);
    tree.fit(data);
    // auto lcs{Cart::Loss::_consecutive_lcs(tree.get_internal_nodes())};
    // for(const auto& lc : lcs) {
    //     std::cout << "New iteration:\n";
    //     for(auto [gamma, LC] : lc)
    //         std::cout << "\t(" << gamma << ", " << LC << ")\n";
    // }
    return 0;
}
