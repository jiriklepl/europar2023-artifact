#!/bin/bash -ex
g++ -DHISTO_HAVE_TBB -DHISTO_IMPL=histo_trav_tbbreduce --std=c++20 -pedantic -O3 -I"noarr-structures/include" histo.cpp -o bin_histo -ltbb -ltbbmalloc
./bin_histo "$1" `wc -c < "$1"`
