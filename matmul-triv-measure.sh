#!/bin/bash

mkdir -p "build/matmul/clang++"
mkdir -p "build/matmul/g++"

INCLUDE_OPTION="-Inoarr-structures/include"
CXX_OPTIONS="$INCLUDE_OPTION -std=c++20 -Ofast -flto -Wall -Wextra -pedantic -march=native -mtune=native"

echo "running compilation:" 1>&2

while read -r compiler; do
	while read -r version; do
        output="build/matmul/$compiler/cpu-triv-$version"
        source="matmul/cpu-triv-$version.cpp"

        call="$compiler -o $output \
$CXX_OPTIONS \
-DNDEBUG -DNLOGGING \
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

if [ $# -gt 0 ] && [ "$1" = "just-compile" ]; then
	exit 0
fi

echo "running validation:" 1>&2

OUTPUT="/tmp/matmul-$(uuidgen)"
OUTPUT2="/tmp/matmul-$(uuidgen)"

while read -r input size; do
	echo "$(date +%H:%M:%S:) running validation on '$input' (size: $size):" 1>&2

    if [ -f "$input" ]; then
        find build/matmul -type f -name "*cpu-triv-*" -mindepth 2 | shuf | while read -r file; do
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
			find build/matmul -type f -name "*cpu-triv-*" -mindepth 2 | shuf | while read -r file; do
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
