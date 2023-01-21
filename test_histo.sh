#!/bin/bash -ex
g++ -DHISTO_IMPL=histo_trav_foreach --std=c++17 -pedantic -O3 -I"noarr-structures/include" histo.cpp -o bin_histo
./bin_histo "$1" `wc -c < "$1"`
