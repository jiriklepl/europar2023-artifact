#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cubagmatmul.cu -o bin_cubagmatmul
./bin_cubagmatmul "$1" | md5sum
