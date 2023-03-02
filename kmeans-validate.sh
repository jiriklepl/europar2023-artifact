#!/bin/sh

OUTPUT="/tmp/kmeans-$(uuidgen)"
OUTPUT2="/tmp/kmeans-$(uuidgen)"

while read -r input size clusters dims; do
	find build/kmeans -type f -mindepth 2 | shuf | while read -r file; do
		echo "$(date +%H:%M:%S:)" "$file" "$input" "$size" 1>&2
		"$file" "$size" "$clusters" "$dims" < "$input"  > "$OUTPUT"

		if [ -f "$OUTPUT2" ]; then
			diff "$OUTPUT" "$OUTPUT2" || exit 1
		fi

		mv "$OUTPUT" "$OUTPUT2"
	done || exit 1

	rm "$OUTPUT2"
done <<EOF
build/kmeans/kmeans_7_4_2000 2000 7 4
build/kmeans/kmeans_7_6_2000 2000 7 6
build/kmeans/kmeans_10_3_2000 2000 10 3
EOF
