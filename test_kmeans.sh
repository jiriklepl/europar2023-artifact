#!/bin/bash -ex
g++ --std=c++17 -pedantic -O3 -I"noarr-structures/include" kmeans.cpp -o bin_kmeans
time ./bin_kmeans "$1" "$2" "$3" < "$4"
