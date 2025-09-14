# Cart++

`Cart++` is a C++ implementation of CART that contains:
- [ ] Fairness criteria
- [x] Node-based losses
- [x] Tree-based losses

The library is header-only (and heavily relies on C++20 templates and concepts),
so just include the appropriate files when compiling.

## Python binding

The module `pycart` is a Python binding using `Cart++`.
The binding is written using `Cython`.

## R binding

Will come at some point

## Debug build and sanitizers

By default, this project assumes compiling with `clang` but supports `gcc` as well.
To build `pycart` with sanitizers, add `-fsanitize=address` at compile-time.
To be able to then use the module, prefix the `python3 <file.py>` with:
```
LD_PRELOAD="$(clang -print-file-name=libasan.so) $(clang -print-file-name=libasan.so)"
```

To have ASAN's logs in some log file, use option `ASAN_OPTIONS="log_path=<path>"`

For instance:
```
make pycart.so
LD_PRELOAD="$(clang -print-file-name=libasan.so) $(clang -print-file-name=libasan.so)" ASAN_OPTIONS="log_path=asan_log.txt" python3 test_cpp.py
file asan_log.txt
```

