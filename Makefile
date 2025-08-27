CXX=g++
OPENMPFLAG=-fopenmp
# CXX=clang++
# OPENMPFLAG=-fopenmp=libomp
FLAGS=-std=c++20 -MMD -Wall -Wextra -O3 $(shell ./get_includes.py) -Isrc/
# FLAGS=-std=c++20 -MMD -Wall -Wextra -g $(shell ./get_includes.py) -fsanitize=address
# -static-libasan

all: bin/default pycart.so

bin/default: obj/main.o
	${CXX} ${FLAGS} $< -o $@

obj/%.o: %.cpp
	${CXX} -c ${FLAGS} $< -o $@

pycart.cpp: pycart.pyx
	cythonize $<

pycart.o: pycart.cpp
	${CXX} ${OPENMPFLAG} -MMD -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -fPIC -c ${FLAGS} $< -o $@

%.so: %.o
	${CXX} -shared -pthread -fwrapv -fno-strict-aliasing -static-libasan $< -o $@

-include obj/*.d
-include *.d

clean:
	rm -f obj/*.o
	rm -f obj/*.d
	rm -f *.o
	rm -f *.d
	rm -f pycart.cpp
	rm -f pycart.so

.PHONY: clean
