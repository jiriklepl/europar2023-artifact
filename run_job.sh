#!/bin/sh

MACHINE=w201
GROUP=ffa

rm -rf noarr-structures
rm -rf build/*/clang++
rm -rf build/*/g++
rm -rf build/*/nvcc

make generate -j8 || exit 1
make generate-small -j8 || exit 1

srun -p "$GROUP" -w "$MACHINE" make all NVCC=@: CLANG=@: -Bj16 || exit 1
srun -p "$GROUP" -w "$MACHINE" ./histo-validate.sh || exit 2
srun -p "$GROUP" -w "$MACHINE" ./matmul-validate.sh || exit 3
for _ in $(seq 10); do
	srun -p "$GROUP" -w "$MACHINE" ./matmul-measure.sh || exit 4
	srun -p "$GROUP" -w "$MACHINE" ./histo-measure.sh || exit 5
done
