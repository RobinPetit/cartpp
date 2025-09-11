#define BRT(F, L) Cart::Regression::BaseRegressionTree<F, Cart::Loss::L<F>>

static inline void CALL_FIT_TREE(void* tree, void* dataset, __FloatingPoint fp, __Loss loss) {
#define FIT(F, L) static_cast<BRT(F, L)*>(tree)->fit(*static_cast<Cart::Dataset<F>*>(dataset))
    if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::MSE) {
        FIT(CART_FLOAT32, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::MSE) {
        FIT(CART_FLOAT64, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::POISSON_DEVIANCE) {
        FIT(CART_FLOAT32, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::POISSON_DEVIANCE) {
        FIT(CART_FLOAT64, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::LORENZ) {
        FIT(CART_FLOAT32, LorenzCurveError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::LORENZ) {
        FIT(CART_FLOAT64, LorenzCurveError);
    } else {
        throw std::runtime_error("Wrong loss or dtype");
    }
#undef FIT
}

static inline void CALL_PREDICT_TREE(void* tree, void* X, void* out, int n, int nb_dims, __FloatingPoint fp, __Loss loss) {
#define PREDICT(F, L) do {                                                  \
        Cart::Array<F> _out(static_cast<F*>(out), n, false);     \
        Cart::Array<F> _X(static_cast<F*>(X), n*nb_dims, false); \
        static_cast<BRT(F, L)*>(tree)->predict(_X, _out);        \
        } while(false)
    if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::MSE) {
        PREDICT(CART_FLOAT32, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::MSE) {
        PREDICT(CART_FLOAT64, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::POISSON_DEVIANCE) {
        PREDICT(CART_FLOAT32, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::POISSON_DEVIANCE) {
        PREDICT(CART_FLOAT64, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::LORENZ) {
        PREDICT(CART_FLOAT32, LorenzCurveError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::LORENZ) {
        PREDICT(CART_FLOAT64, LorenzCurveError);
    } else {
        throw std::runtime_error("Wrong loss or dtype");
    }
#undef PREDICT
}

static inline void CALL_CREATE_TREE(void** tree, Cart::TreeConfig* config, __FloatingPoint fp, __Loss loss) {
#define CREATE(F, L) do {                                                   \
        *reinterpret_cast<BRT(F, L)**>(tree) = new BRT(F, L)(*config); \
        } while(false)
    if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::MSE) {
        CREATE(CART_FLOAT32, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::MSE) {
        CREATE(CART_FLOAT64, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::POISSON_DEVIANCE) {
        CREATE(CART_FLOAT32, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::POISSON_DEVIANCE) {
        CREATE(CART_FLOAT64, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::LORENZ) {
        CREATE(CART_FLOAT32, LorenzCurveError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::LORENZ) {
        CREATE(CART_FLOAT64, LorenzCurveError);
    } else {
        throw std::runtime_error("Wrong loss or dtype");
    }
#undef CREATE
}

static inline void CALL_DELETE_TREE(void* tree, __FloatingPoint fp, __Loss loss) {
#define DELETE(F, L) delete static_cast<BRT(F, L)*>(tree)
    if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::MSE) {
        DELETE(CART_FLOAT32, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::MSE) {
        DELETE(CART_FLOAT64, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::POISSON_DEVIANCE) {
        DELETE(CART_FLOAT32, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::POISSON_DEVIANCE) {
        DELETE(CART_FLOAT64, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::LORENZ) {
        DELETE(CART_FLOAT32, LorenzCurveError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::LORENZ) {
        DELETE(CART_FLOAT64, LorenzCurveError);
    } else {
        throw std::runtime_error("Wrong loss or dtype");
    }
#undef DELETE
}

static inline void CALL_GET_NB_INTERNAL_NODES_TREE(void* tree, size_t* size, __FloatingPoint fp, __Loss loss) {
#define GET_NB_INTERNAL_NODES(F, L) do {                                                                 \
            *size = static_cast<BRT(F, L)*>(tree)->get_internal_nodes().size(); \
        } while(false)
    if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::MSE) {
        GET_NB_INTERNAL_NODES(CART_FLOAT32, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::MSE) {
        GET_NB_INTERNAL_NODES(CART_FLOAT64, MeanSquaredError);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::POISSON_DEVIANCE) {
        GET_NB_INTERNAL_NODES(CART_FLOAT32, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::POISSON_DEVIANCE) {
        GET_NB_INTERNAL_NODES(CART_FLOAT64, PoissonDeviance);
    } else if(fp == __FloatingPoint::FLOAT32 and loss == __Loss::LORENZ) {
        GET_NB_INTERNAL_NODES(CART_FLOAT32, LorenzCurveError);
    } else if(fp == __FloatingPoint::FLOAT64 and loss == __Loss::LORENZ) {
        GET_NB_INTERNAL_NODES(CART_FLOAT64, LorenzCurveError);
    } else {
        throw std::runtime_error("Wrong loss or dtype");
    }
#undef GET_NB_INTERNAL_NODES
}

#undef BRT
