#!/bin/sh

OUTPUT=out/matmul-$(uname -n)-$(uuidgen).csv
REPS=10

mkdir -p out

while read -r input size; do
    for _ in $(seq "$REPS"); do
        find build/matmul -mindepth 2 | shuf | while read -r file; do
            printf "%s" "$file,$input,$size,$(uname -n),$(date)," >> "$OUTPUT"

            "$file" "$input" "$size" > /dev/null 2>> "$OUTPUT"
        done
    done
done <<EOF
build/matmul/matrices_1024 1024
build/matmul/matrices_2048 2048
build/matmul/matrices_4096 4096
build/matmul/matrices_8192 8192
EOF
