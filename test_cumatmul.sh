#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cumatmul.cu -o bin_cumatmul
./bin_cumatmul "$1" | md5sum
