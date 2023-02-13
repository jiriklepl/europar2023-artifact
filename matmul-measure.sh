#!/bin/sh

OUTPUT=out/matmul-$(uname -n)-$(uuidgen).csv
REPS=10

mkdir -p out

while read -r input size; do
    for _ in $(seq "$REPS"); do
        find build/matmul -mindepth 2 | while read -r file; do
            printf "%s" "$file,$input,$size,$(uname -n),$(date)," >> "$OUTPUT"

            "$file" "$input" "$size" > /dev/null 2>> "$OUTPUT"
        done
    done
done <<EOF
build/matmul/matrices_64 64
build/matmul/matrices_128 128
build/matmul/matrices_256 256
build/matmul/matrices_512 512
build/matmul/matrices_1024 1024
EOF
