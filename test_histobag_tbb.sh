#!/bin/bash -ex
g++ -DHISTO_HAVE_TBB -DHISTO_IMPL=histo_trav_tbbreduce --std=c++20 -pedantic -O3 -I"noarr-structures/include" histobag.cpp -o bin_histobag -ltbb -ltbbmalloc
./bin_histobag "$1" `wc -c < "$1"`
