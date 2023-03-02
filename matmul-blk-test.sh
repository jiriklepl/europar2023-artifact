#!/bin/bash

mkdir -p "tests/matmul/clang++"
mkdir -p "tests/matmul/g++"

INCLUDE_OPTION="-Inoarr-structures/include"
CXX_OPTIONS="$INCLUDE_OPTION -std=c++20 -Ofast -Werror -flto -Wall -Wextra -pedantic -march=native -mtune=native"

echo "running compilation:" 1>&2

while read -r compiler; do
	while read -r version; do
		for block_order in $(seq 0 5); do
			for dim_order in $(seq 0 1); do
				output="tests/matmul/$compiler/cpu-blk_${block_order}_${dim_order}-$version"
				source="matmul/cpu-blk-$version.cpp"

				call="$compiler -o $output \
$CXX_OPTIONS \
-DDEBUG -DLOGGING \
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

echo "running tests:" 1>&2

OUTPUT="/tmp/matmul-$(uuidgen)"
OUTPUT2="/tmp/matmul-$(uuidgen)"
LOG="/tmp/matmul-$(uuidgen)"
LOG2="/tmp/matmul-$(uuidgen)"

while read -r input size; do
	echo "$(date +%H:%M:%S:) running tests on '$input' (size: $size):" 1>&2

	if [ -f "$input" ]; then
		for block_order in $(seq 0 5); do
			for dim_order in $(seq 0 1); do
				find tests/matmul -type f -name "*cpu-blk_${block_order}_${dim_order}-*" -mindepth 2 | shuf | while read -r file; do
					echo "$(date +%H:%M:%S:)" "$file" "$input" "$size" 1>&2
					"$file" "$input" "$size" > "$OUTPUT" 2> "$LOG"

					if [ -f "$OUTPUT2" ]; then
						diff -u "$OUTPUT" "$OUTPUT2" || exit 1
					fi

					if [ -f "$LOG2" ]; then
						diff -u <(awk 'NR>1{print prev} {prev=$0}' "$LOG") <(awk 'NR>1{print prev} {prev=$0}' "$LOG2") || exit 1
					fi

					mv "$OUTPUT" "$OUTPUT2"
					mv "$LOG" "$LOG2"
				done || exit 1

				rm "$LOG2"
			done
		done

		rm "$OUTPUT2"
	else
		echo "warning: requesting tests on nonexistent input" 1>&2
	fi
done <<EOF
build/matmul/matrices_64 64
EOF
