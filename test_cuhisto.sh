#!/bin/bash -ex
nvcc -O3 -I noarr-structures/include -allow-unsupported-compiler --expt-relaxed-constexpr cuhisto.cu -o bin_cuhisto
./bin_cuhisto "$1" `wc -c < "$1"`
