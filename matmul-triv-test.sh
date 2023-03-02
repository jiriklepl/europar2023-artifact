#!/bin/bash

mkdir -p "tests/matmul/clang++"
mkdir -p "tests/matmul/g++"

INCLUDE_OPTION="-Inoarr-structures/include"
CXX_OPTIONS="$INCLUDE_OPTION -std=c++20 -Ofast -Werror -flto -Wall -Wextra -pedantic -march=native -mtune=native"

echo "running compilation:" 1>&2

while read -r compiler; do
	while read -r version; do
        output="tests/matmul/$compiler/cpu-triv-$version"
        source="matmul/cpu-triv-$version.cpp"

        call="$compiler -o $output \
$CXX_OPTIONS \
-DDEBUG -DLOGGING \
$source \
-DA_ROW -DB_ROW -DC_ROW"
        if [ "$source" -nt "$output" ]; then
            echo "$call" 1>&2
            $call
        fi
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
        find tests/matmul -type f -name "*cpu-triv-*" -mindepth 2 | shuf | while read -r file; do
            echo "$(date +%H:%M:%S:)" "$file" "$input" "$size" 1>&2
            "$file" "$input" "$size" > "$OUTPUT" 2> "$LOG"

            if [ -f "$OUTPUT2" ]; then
                diff -u "$OUTPUT" "$OUTPUT2" || exit 1
                diff -u <(awk 'NR>1{print prev} {prev=$0}' "$LOG") <(awk 'NR>1{print prev} {prev=$0}' "$LOG2") || exit 1
            fi

            mv "$OUTPUT" "$OUTPUT2"
            mv "$LOG" "$LOG2"
        done || exit 1

        rm "$OUTPUT2"
        rm "$LOG2"
    else
        echo "warning: requesting tests on nonexistent input" 1>&2
    fi
done <<EOF
build/matmul/matrices_64 64
EOF
