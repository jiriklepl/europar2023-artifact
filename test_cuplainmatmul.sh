#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cuplainmatmul.cu -o bin_cuplainmatmul -lcublas
./bin_cuplainmatmul "$1" | md5sum
