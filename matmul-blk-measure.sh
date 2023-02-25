#!/bin/bash

mkdir -p "build/matmul/clang++"
mkdir -p "build/matmul/g++"

INCLUDE_OPTION="-Inoarr-structures/include"
CXX_OPTIONS="$INCLUDE_OPTION -std=c++20 -Ofast -flto -Wall -Wextra -pedantic -march=native -mtune=native"

echo "running compilation:" 1>&2

while read -r compiler; do
	while read -r version; do
		for block_order in $(seq 0 5); do
			for dim_order in $(seq 0 1); do
				output="build/matmul/$compiler/cpu-blk_${block_order}_${dim_order}-$version"
				source="matmul/cpu-blk-$version.cpp"

				call="$compiler -o $output \
$CXX_OPTIONS \
-DNDEBUG -DNLOGGING \
$source \
-DA_ROW -DB_ROW -DC_ROW \
-DBLOCK_I -DBLOCK_J -DBLOCK_K \
-DBLOCK_SIZE=16 -DBLOCK_ORDER=$block_order -DDIM_ORDER=$dim_order"
				if [ "$source" -nt "$output" ]; then
					echo "$call" 1>&2
					$call
				fi
			done
		done
	done <<EOF
noarr
noarr-bag
policy
EOF
done <<EOF
g++
clang++
EOF

if [ $# -gt 0 ] && [ "$1" = "just-compile" ]; then
	exit 0
fi

echo "running validation:" 1>&2

OUTPUT="/tmp/matmul-$(uuidgen)"
OUTPUT2="/tmp/matmul-$(uuidgen)"

while read -r input size; do
	echo "$(date +%H:%M:%S:) running validation on '$input' (size: $size):" 1>&2

	if [ -f "$input" ]; then
		find build/matmul -type f -name "*cpu-blk_${block_order}_${dim_order}-*" -mindepth 2 | shuf | while read -r file; do
			echo "$(date +%H:%M:%S:)" "$file" "$input" "$size" 1>&2
			"$file" "$input" "$size" > "$OUTPUT"

			if [ -f "$OUTPUT2" ]; then
				diff -u "$OUTPUT" "$OUTPUT2" || exit 1
			fi

			mv "$OUTPUT" "$OUTPUT2"
		done || exit 1

		rm "$OUTPUT2"
	else
		echo "warning: requesting validation on nonexistent input" 1>&2
	fi
done <<EOF
build/matmul/matrices_64 64
EOF

if [ $# -gt 0 ] && [ "$1" = "no-measure" ]; then
	exit 0
fi

OUTPUT=out/matmul-$(uname -n)-$(uuidgen).csv
REPS=10

mkdir -p "out"

echo "running measurements (outputting to '$OUTPUT'):" 1>&2

while read -r input size; do
	echo "$(date +%H:%M:%S:) running measurements on '$input' (size: $size):" 1>&2

	if [ -f "$input" ]; then
		for _ in $(seq "$REPS"); do
			find build/matmul -type f -name "*cpu-blk_[0-9]_[0-9]-*" -mindepth 2 | shuf | while read -r file; do
				printf "%s" "$file,$input,$size,$(uname -n),$(date)," >> "$OUTPUT"
				"$file" "$input" "$size" > /dev/null 2>> "$OUTPUT" || exit 1
			done || exit 1
		done || exit 1
	else
		echo "warning: requesting measurements on nonexistent input" 1>&2
	fi
done <<EOF
build/matmul/matrices_64 64
build/matmul/matrices_128 128
build/matmul/matrices_256 256
build/matmul/matrices_512 512
build/matmul/matrices_1024 1024
EOF
