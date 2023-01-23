#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cusmatmul1.cu -o bin_cusmatmul1
./bin_cusmatmul1 "$1" | md5sum
