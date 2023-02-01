#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cuhistobag.cu -o bin_cuhistobag
./bin_cuhistobag "$1" `wc -c < "$1"`
