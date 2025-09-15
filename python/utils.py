import os.path
from pathlib import Path

LOSSES = {
    'MSE': 'MeanSquaredError',
    'POISSON_DEVIANCE': 'PoissonDeviance',
    'NON_CROSSING_LORENZ': 'NonCrossingLorenzCurveError',
    'CROSSING_LORENZ': 'CrossingLorenzCurveError'
}
DTYPES = {
    'FLOAT32': 'CART_FLOAT32',
    'FLOAT64': 'CART_FLOAT64'
}


FUNCTION_ARGS = [
    (
        'FIT',
        'static_cast<BRT(F, L)*>(tree)->fit(*static_cast<Cart::Dataset<F>*>(dataset))',
        'void* tree, void* dataset'
    ),
    (
        'PREDICT',
        '''do {                                                  \\
        Cart::Array<F> _out(static_cast<F*>(out), n, false);     \\
        Cart::Array<F> _X(static_cast<F*>(X), n*nb_dims, false); \\
        static_cast<BRT(F, L)*>(tree)->predict(_X, _out);        \\
        } while(false)''',
        'void* tree, void* X, void* out, int n, int nb_dims'
    ),
    (
        'CREATE',
        '''do {                                                   \\
        *reinterpret_cast<BRT(F, L)**>(tree) = new BRT(F, L)(*config); \\
        } while(false)''',
        'void** tree, Cart::TreeConfig* config'
    ),
    (
        'DELETE',
        'delete static_cast<BRT(F, L)*>(tree)',
        'void* tree'
    ),
    (
        'GET_NB_INTERNAL_NODES',
        '''do {                                                                 \\
            *size = static_cast<BRT(F, L)*>(tree)->get_internal_nodes().size(); \\
        } while(false)''',
        'void* tree, size_t* size'
    ),
    (
        'GET_FEATURE_IMPORTANCE',
        'static_cast<BRT(F, L)*>(tree)->get_feature_importance(static_cast<F*>(array))',
        'void* tree, void* array'
    ),
    (
        'GET_INTERNAL_NODES',
        '*reinterpret_cast<std::vector<Cart::Node<F>*>**>(ret) = const_cast<std::vector<Cart::Node<F>*>*>(&static_cast<BRT(F, L)*>(tree)->get_internal_nodes())',
        'void* tree, void** ret'
    ),
    (
        'GET_ROOT',
        '*ret = static_cast<void*>(const_cast<Cart::Node<F>*>(static_cast<BRT(F, L)*>(tree)->get_root()))',
        'void* tree, void** ret'
    )
]


def make_function(def_name, def_value, args='void* tree'):
    function = []
    function.append(f'static inline void CALL_{def_name}_TREE({args}, __FloatingPoint fp, __Loss loss) {{')
    function.append(f'#define {def_name}(F, L) {def_value}')
    first = True
    for loss in LOSSES.keys():
        for dtype in DTYPES.keys():
            if first:
                first = False
                IF = 'if'
            else:
                IF = '} else if'
            function.append(f'    {IF}(fp == __FloatingPoint::{dtype} and loss == __Loss::{loss}) {{')
            function.append(f'        {def_name}({DTYPES[dtype]}, {LOSSES[loss]});')
    function.append('    } else {')
    function.append('        throw std::runtime_error("Wrong loss or dtype");')
    function.append('    }')
    function.append(f'#undef {def_name}')
    function.append('}')
    return function

def make_cpp_wrapper():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(Path(dirname) / '__pycart_calls.hpp', 'w') as f:
        f.write('#define BRT(F, L) Cart::Regression::BaseRegressionTree<F, Cart::Loss::L<F>>\n\n')
        for args in FUNCTION_ARGS:
            f.write('\n'.join(make_function(*args)))
            f.write('\n\n')
        f.write('#undef BRT\n')


if __name__ == '__main__':
    make_cpp_wrapper()
