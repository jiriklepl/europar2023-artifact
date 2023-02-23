#!/bin/sh

OUTPUT="/tmp/matmul-$(uuidgen)"
OUTPUT2="/tmp/matmul-$(uuidgen)"

while read -r input size; do
	find build/matmul -type f -mindepth 2 | shuf | while read -r file; do
		echo "$(date +%H:%M:%S:)" "$file" "$input" "$size" 1>&2
		"$file" "$input" "$size" > "$OUTPUT"

		if [ -f "$OUTPUT2" ]; then
			diff "$OUTPUT" "$OUTPUT2" || exit 1
		fi

		mv "$OUTPUT" "$OUTPUT2"
	done || exit 1

	rm "$OUTPUT2"
done <<EOF
build/matmul/matrices_1024 1024
EOF
