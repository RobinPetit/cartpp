#include "config.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "tree.hpp"

typedef double Float;
int main() {
    auto data{Cart::Dataset<double>::load_from("./dataset_filtered_cat.cartbin")};
    Cart::TreeConfig config;
    config.interaction_depth = 50;
    config.nb_covariates = 100;
    config.split_type = Cart::NodeSelector::BEST_FIRST;
    config.verbose = true;
    using Loss = Cart::Loss::NonCrossingLorenzCurveError<double>;
    Cart::Regression::BaseRegressionTree<double, Loss> tree(config);
    tree.fit(data);
    return 0;
}
