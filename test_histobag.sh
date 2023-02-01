#!/bin/bash -ex
g++ -DHISTO_IMPL=histo_trav_foreach --std=c++17 -pedantic -O3 -I"noarr-structures/include" histobag.cpp -o bin_histobag
./bin_histobag "$1" `wc -c < "$1"`
