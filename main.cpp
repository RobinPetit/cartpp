#include <iostream>

#include "config.hpp"
#include "dataset.hpp"

typedef double Float;
int main() {
    auto data{Cart::Dataset<double>::load_from("../../dataset_numerical.cartbin")};
    Cart::TreeConfig config;
    return 0;
}
