# CXX=g++
# OPENMPFLAG=-fopenmp
CXX=clang++
OPENMPFLAG=-fopenmp=libomp
FLAGS=-std=c++20 -Isrc/ $(shell ./get_includes.py) -Wall -Wextra -Wdisabled-optimization -Wundef -Wpedantic -MMD

CXXFLAGS=-O3 ${FLAGS} -DNDEBUG
#-Rpass-analysis=loop-vectorize
# CXXFLAGS=-g -pg -O0 ${FLAGS}
# -fsanitize=address
# -static-libasan

LINK_FLAGS=-fno-strict-aliasing
# -fsanitize=address
#
DOC_DIR=doc/cartpp
REMOTE_ORIGIN=git@github.com:RobinPetit/cartpp.git
SRC=$(wildcard src/*.hpp)

all: pycart.so

obj/%.o: %.cpp
	${CXX} -c ${CXXFLAGS} $< -o $@

python/__pycart_calls.hpp: python/utils.py
	python3 $<

python/pycart.cpp: python/pycart.pyx
	cythonize $<

obj/pycart.o: python/pycart.cpp
	${CXX} ${OPENMPFLAG} -MMD -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -fPIC -c ${CXXFLAGS} $< -o $@

%.so: obj/%.o
	${CXX} -fPIC -shared -pthread -fwrapv ${LINK_FLAGS} $< -o $@

-include obj/*.d
-include *.d

doc: ${SRC} Doxyfile
	doxygen Doxyfile
	touch ${DOC_DIR}/html/.nojekyll

pushdoc: doc
	cd ${DOC_DIR}/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && git commit -m "Build the doc" && git push -f origin HEAD:gh-pages

clean:
	rm -f bin/*
	rm -f obj/*.o
	rm -f obj/*.d
	rm -f *.so
	rm -f python/pycart.cpp

.PHONY: doc pushdoc clean
